import torch
import data_hyperparameters
import os
from abc import ABC
import matplotlib.pyplot as plt
import datetime
from math import nan, log
from model_pipeline import average_bleu

if not os.path.exists('learning_curves/'):
    os.mkdir('learning_curves/')

device = torch.device('cuda' if data_hyperparameters.USE_CUDA else 'cpu')
LARGE_NEGATIVE = -1e9


class BaseModelClass(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.train_bleus = []
        self.valid_bleus = []
        self.num_epochs_trained = 0
        self.latest_scheduled_lr = None
        self.lr_history = []
        self.train_time = 0.
        self.num_trainable_params = 0
        self.instantiated = datetime.datetime.now()
        self.name = ''
        self.batch_size = data_hyperparameters.BATCH_SIZE
        self.teacher_forcing_proportion = data_hyperparameters.TEACHER_FORCING_PROPORTION_START
        self.teacher_forcing_proportion_history = []

    def count_parameters(self):
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def finish_setup(self):
        self.count_parameters()

    def get_model_performance_data(self, train_dataloader, valid_dataloader, test_dataloader):
        final_train_loss = nan if len(self.train_losses) == 0 else self.train_losses[-1]
        final_valid_loss = nan if len(self.valid_losses) == 0 else self.valid_losses[-1]
        final_train_bleu = average_bleu(train_dataloader, self)
        final_valid_bleu = average_bleu(valid_dataloader, self)
        final_test_bleu = average_bleu(test_dataloader, self)
        average_time_per_epoch = nan if self.num_epochs_trained == 0 else self.train_time / self.num_epochs_trained
        model_data = {'name': self.name, 'total_train_time': self.train_time, 'final_train_bleu': final_train_bleu,
                      'final_valid_bleu': final_valid_bleu, 'final_test_bleu': final_test_bleu,
                      'num_epochs': self.num_epochs_trained, 'trainable_params': self.num_trainable_params,
                      'final_train_loss': final_train_loss, 'final_valid_loss': final_valid_loss,
                      'model_created': self.instantiated, 'average_time_per_epoch': average_time_per_epoch,
                      'batch_size': self.batch_size}
        return model_data

    def plot_losses(self, include_lrs=True):
        fig, ax = plt.subplots()
        ax.plot(range(self.num_epochs_trained), self.train_losses, label='Training')
        ax.plot(range(self.num_epochs_trained), self.valid_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning curve for {0}'.format(self.name))
        ax.legend()
        plt.savefig('learning_curves/learning_curve_{0}.png'.format(self.name))

        fig, ax = plt.subplots()
        ax.plot(range(self.num_epochs_trained), self.teacher_forcing_proportion_history,
                label='Teacher forcing proportion')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Proportion')
        ax.set_title('Teacher forcing proportions for {0}'.format(self.name))
        ax.legend()
        plt.savefig('learning_curves/teacher_forcing_proportions_{0}.png'.format(self.name))

        fig, ax = plt.subplots()
        ax.scatter(range(self.num_epochs_trained), self.train_bleus, label='Training')
        ax.scatter(range(self.num_epochs_trained), self.valid_bleus, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BLEU')
        ax.set_title('BLEU scores for {0}'.format(self.name))
        ax.legend()
        plt.savefig('learning_curves/bleus_{0}.png'.format(self.name))

        if include_lrs:
            fig, ax = plt.subplots()
            ax.plot(range(self.num_epochs_trained), self.lr_history, label='Learning rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.set_title('Learning rates for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/learning_rates_{0}.png'.format(self.name))


class EncoderGRU(BaseModelClass):
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, use_packing, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, bidirectional=bidirectional, input_size=embedding_dimension,
                                hidden_size=hidden_size, batch_first=True,
                                dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.use_packing = use_packing
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        '''
        Take encoder inputs to the output and final hidden states of the encoder
        :param inputs: Tensor of shape [B, T] (type long)
        :return: encoder_outputs, encoder_hidden
        encoder_output: Tensor of shape [B, T, DH] where D = num_directions
        encoder_hidden: Tensor of shape [L, B, DH] where D = num_directions
        '''
        embeds = self.embedding_dropout(self.embedding(inputs))  # [B, T, E]
        if self.use_packing:
            input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)  # [B]
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False,
                                                             batch_first=True)
        encoder_output, encoder_hidden = self.GRU(embeds)  # [B, T, DH], [LD, B, H]
        if self.use_packing:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)
        if not self.bidirectional:
            return encoder_output, encoder_hidden
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, -1, self.hidden_size)  # [L, D=2, B, H]
        encoder_hidden = encoder_hidden.transpose(1, 2)  # [L, B, D=2, H]
        encoder_hidden_forward = encoder_hidden[:, :, 0, :]  # [L, B, H]
        encoder_hidden_backward = encoder_hidden[:, :, 1, :]  # [L, B, H]
        return encoder_output, torch.cat([encoder_hidden_forward, encoder_hidden_backward], dim=2)


class DecoderGRU(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size,
                                batch_first=True, dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)
        self.name = name
        self.decoder_type = 'vanilla'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, decoder_output, decoder_hidden):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :return: log_probs, gru_output, gru_hidden
        log_probs: Tensor of shape [B, V]
        gru_output: Tensor of shape [B, S=1, H]
        gru_hidden: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        gru_output, gru_hidden = self.GRU(embeds, decoder_hidden)  # [B, S=1, H], [L, B, H]
        proj = self.linear(gru_output.squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, gru_output, gru_hidden


class DecoderGRUWithContext(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                hidden_size=hidden_size, batch_first=True,
                                dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'context'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, decoder_output, decoder_hidden):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :return: log_probs, gru_output, gru_hidden
        log_probs: Tensor of shape [B, V]
        gru_output: Tensor of shape [B, S=1, H]
        gru_hidden: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        context = encoder_hidden[-1, :, :].unsqueeze(1)  # [B, 1, H]
        gru_output, gru_hidden = self.GRU(torch.cat([embeds, context], dim=2), decoder_hidden)  # [B, S=1, H], [L, B, H]
        proj = self.linear(torch.cat([gru_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, gru_output, gru_hidden


class DecoderGRUWithDotProductAttention(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                hidden_size=hidden_size, batch_first=True,
                                dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'dot_product_attention'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, decoder_output, decoder_hidden):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :return: log_probs, gru_output, gru_hidden
        log_probs: Tensor of shape [B, V]
        gru_output: Tensor of shape [B, S=1, H]
        gru_hidden: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        similarity_scores = torch.bmm(encoder_output, decoder_output.transpose(1, 2))  # [B, T, 1]
        similarity_scores = similarity_scores.masked_fill(attention_mask.unsqueeze(2), LARGE_NEGATIVE)  # [B, T, 1]
        attention_weights = torch.nn.functional.softmax(similarity_scores, dim=1).transpose(1, 2)  # [B, 1, T]
        context = torch.bmm(attention_weights, encoder_output)  # [B, 1, H]
        gru_output, gru_hidden = self.GRU(torch.cat([embeds, context], dim=2), decoder_hidden)  # [B, S=1, H], [L, B, H]
        proj = self.linear(torch.cat([gru_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, gru_output, gru_hidden


class DecoderGRUWithLearnableAttention(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.attention_layer_1 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.attention_layer_2 = torch.nn.Linear(hidden_size, 1)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                hidden_size=hidden_size, batch_first=True,
                                dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'learnable_attention'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, decoder_output, decoder_hidden):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :return: log_probs, gru_output, gru_hidden
        log_probs: Tensor of shape [B, V]
        gru_output: Tensor of shape [B, S=1, H]
        gru_hidden: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        source_length = encoder_output.shape[1]
        attention_layer_1 = self.attention_layer_1(
            torch.cat([encoder_output, decoder_output.repeat(1, source_length, 1)], dim=2))  # [B, T, H]
        attention_layer_2 = self.attention_layer_2(torch.tanh(attention_layer_1))  # [B, T, 1]
        attention_layer_2 = attention_layer_2.masked_fill(attention_mask.unsqueeze(2), LARGE_NEGATIVE)  # [B, T, 1]
        attention_weights = torch.nn.functional.softmax(attention_layer_2, dim=1).transpose(1, 2)  # [B, 1, T]
        context = torch.bmm(attention_weights, encoder_output)  # [B, 1, H]
        gru_output, gru_hidden = self.GRU(torch.cat([embeds, context], dim=2), decoder_hidden)  # [B, S=1, H], [L, B, H]
        proj = self.linear(torch.cat([gru_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, gru_output, gru_hidden


class EncoderDecoderGRU(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers, decoder_type='vanilla', bidirectional_encoder=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 encoder_embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 decoder_embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 encoder_embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 decoder_embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 encoder_inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 decoder_inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 encoder_intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 decoder_intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 use_packing=True, name='GRU', encoder_name='GRU', decoder_name='GRU'):
        super().__init__()
        self.encoder = EncoderGRU(input_lang, num_layers, bidirectional_encoder, hidden_size,
                                  encoder_embedding_dimension, encoder_embedding_dropout,
                                  encoder_inter_recurrent_layer_dropout, encoder_intra_recurrent_layer_dropout,
                                  use_packing, encoder_name)
        self.decoder_hidden_size = 2 * hidden_size if bidirectional_encoder else hidden_size
        self.decoder_type = decoder_type
        self.output_vocab = output_lang.n_words
        if decoder_type == 'vanilla':
            self.decoder = DecoderGRU(output_lang, num_layers, self.decoder_hidden_size, decoder_embedding_dimension,
                                      decoder_embedding_dropout, decoder_inter_recurrent_layer_dropout,
                                      decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'context':
            self.decoder = DecoderGRUWithContext(output_lang, num_layers, self.decoder_hidden_size,
                                                 decoder_embedding_dimension, decoder_embedding_dropout,
                                                 decoder_inter_recurrent_layer_dropout,
                                                 decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'dot_product_attention':
            self.decoder = DecoderGRUWithDotProductAttention(output_lang, num_layers, self.decoder_hidden_size,
                                                             decoder_embedding_dimension, decoder_embedding_dropout,
                                                             decoder_inter_recurrent_layer_dropout,
                                                             decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'learnable_attention':
            self.decoder = DecoderGRUWithLearnableAttention(output_lang, num_layers, self.decoder_hidden_size,
                                                            decoder_embedding_dimension, decoder_embedding_dropout,
                                                            decoder_inter_recurrent_layer_dropout,
                                                            decoder_intra_recurrent_layer_dropout, decoder_name)
        else:
            raise Exception('Decoder type {0} not supported'.format(decoder_type))
        self.name = name + '_' + decoder_type + '_decoder'
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        '''
        Pass through the encoder decoder stack
        :param inputs: Tensor of shape [B, T] (type long)
        :param outputs: Tensor of shape [B, S] (type long)
        :param teacher_force: bool
        :return: all_log_probs
        all_log_probs: Tensor of shape [S, B, V]
        '''
        batch_size, target_length = outputs.shape
        all_log_probs = torch.zeros(target_length, batch_size, self.output_vocab, device=device)  # [S, B, V]
        decoder_input = outputs[:, 0]  # [B]
        encoder_output, encoder_hidden = self.encoder(inputs)  # [B, T, H], [L, B, H]
        attention_mask = inputs == data_hyperparameters.PAD_TOKEN  # [B, T]
        decoder_output = encoder_output[:, -1, :].unsqueeze(1)  # [B, 1, H]
        decoder_hidden = encoder_hidden
        for t in range(1, target_length):
            log_probs, decoder_output, decoder_hidden = self.decoder(decoder_input, attention_mask, encoder_output,
                                                                     encoder_hidden, decoder_output,
                                                                     decoder_hidden)  # [B, V], [B, 1, H], [L, B, H]
            all_log_probs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else torch.argmax(log_probs, dim=1)
        return all_log_probs


class EncoderLSTM(BaseModelClass):
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, use_packing, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(num_layers=num_layers, bidirectional=bidirectional, input_size=embedding_dimension,
                                  hidden_size=hidden_size, batch_first=True,
                                  dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.use_packing = use_packing
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        '''
        Take encoder inputs to the output and final hidden states of the encoder
        :param inputs: Tensor of shape [B, T] (type long)
        :return: encoder_outputs, encoder_hidden, encoder_cell
        encoder_output: Tensor of shape [B, T, DH] where D = num_directions
        encoder_hidden: Tensor of shape [L, B, DH] where D = num_directions
        encoder_cell: Tensor of shape [L, B, DH] where D = num_directions
        '''
        embeds = self.embedding_dropout(self.embedding(inputs))  # [B, T, E]
        if self.use_packing:
            input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)  # [B]
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False,
                                                             batch_first=True)
        encoder_output, (encoder_hidden, encoder_cell) = self.LSTM(embeds)  # [B, T, DH], [LD, B, H], [LD, B, H]
        if self.use_packing:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)
        if not self.bidirectional:
            return encoder_output, encoder_hidden, encoder_cell
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, -1, self.hidden_size)  # [L, D=2, B, H]
        encoder_hidden = encoder_hidden.transpose(1, 2)  # [L, B, D=2, H]
        encoder_hidden_forward = encoder_hidden[:, :, 0, :]  # [L, B, H]
        encoder_hidden_backward = encoder_hidden[:, :, 1, :]  # [L, B, H]
        encoder_cell = encoder_cell.view(self.num_layers, 2, -1, self.hidden_size)  # [L, D=2, B, H]
        encoder_cell = encoder_cell.transpose(1, 2)  # [L, B, D=2, H]
        encoder_cell_forward = encoder_cell[:, :, 0, :]  # [L, B, H]
        encoder_cell_backward = encoder_cell[:, :, 1, :]  # [L, B, H]
        return encoder_output, torch.cat([encoder_hidden_forward, encoder_hidden_backward], dim=2), \
               torch.cat([encoder_cell_forward, encoder_cell_backward], dim=2)


class DecoderLSTM(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size,
                                  batch_first=True, dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)
        self.name = name
        self.decoder_type = 'vanilla'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, encoder_cell, decoder_output,
                decoder_hidden, decoder_cell):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param encoder_cell: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :param decoder_cell: Tensor of shape [L, B, H]
        :return: log_probs, lstm_output, lstm_hidden, lstm_cell
        log_probs: Tensor of shape [B, V]
        lstm_output: Tensor of shape [B, S=1, H]
        lstm_hidden: Tensor of shape [L, B, H]
        lstm_cell: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        lstm_output, (lstm_hidden, lstm_cell) = self.LSTM(embeds, (
        decoder_hidden, decoder_cell))  # [B, S=1, H], [L, B, H], [L, B, H]
        proj = self.linear(lstm_output.squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, lstm_output, lstm_hidden, lstm_cell


class DecoderLSTMWithContext(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                  hidden_size=hidden_size, batch_first=True,
                                  dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'context'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, encoder_cell, decoder_output,
                decoder_hidden, decoder_cell):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param encoder_cell: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :param decoder_cell: Tensor of shape [L, B, H]
        :return: log_probs, lstm_output, lstm_hidden, lstm_cell
        log_probs: Tensor of shape [B, V]
        lstm_output: Tensor of shape [B, S=1, H]
        lstm_hidden: Tensor of shape [L, B, H]
        lstm_cell: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        context = encoder_hidden[-1, :, :].unsqueeze(1)  # [B, 1, H]
        lstm_output, (lstm_hidden, lstm_cell) = self.LSTM(torch.cat([embeds, context], dim=2), (
        decoder_hidden, decoder_cell))  # [B, S=1, H], [L, B, H], [L, B, H]
        proj = self.linear(torch.cat([lstm_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, lstm_output, lstm_hidden, lstm_cell


class DecoderLSTMWithDotProductAttention(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                  hidden_size=hidden_size, batch_first=True,
                                  dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'dot_product_attention'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, encoder_cell, decoder_output,
                decoder_hidden, decoder_cell):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param encoder_cell: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :param decoder_cell: Tensor of shape [L, B, H]
        :return: log_probs, lstm_output, lstm_hidden, lstm_cell
        log_probs: Tensor of shape [B, V]
        lstm_output: Tensor of shape [B, S=1, H]
        lstm_hidden: Tensor of shape [L, B, H]
        lstm_cell: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        similarity_scores = torch.bmm(encoder_output, decoder_output.transpose(1, 2))  # [B, T, 1]
        similarity_scores = similarity_scores.masked_fill(attention_mask.unsqueeze(2), LARGE_NEGATIVE)  # [B, T, 1]
        attention_weights = torch.nn.functional.softmax(similarity_scores, dim=1).transpose(1, 2)  # [B, 1, T]
        context = torch.bmm(attention_weights, encoder_output)  # [B, 1, H]
        lstm_output, (lstm_hidden, lstm_cell) = self.LSTM(torch.cat([embeds, context], dim=2), (
        decoder_hidden, decoder_cell))  # [B, S=1, H], [L, B, H], [L, B, H]
        proj = self.linear(torch.cat([lstm_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, lstm_output, lstm_hidden, lstm_cell


class DecoderLSTMWithLearnableAttention(BaseModelClass):
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.lang_name = lang.name
        self.vocab_size = lang.n_words
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.attention_layer_1 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.attention_layer_2 = torch.nn.Linear(hidden_size, 1)
        self.LSTM = torch.nn.LSTM(num_layers=num_layers, input_size=embedding_dimension + hidden_size,
                                  hidden_size=hidden_size, batch_first=True,
                                  dropout=inter_recurrent_layer_dropout if num_layers > 1 else 0.)
        self.linear = torch.nn.Linear(2 * hidden_size + embedding_dimension, self.vocab_size)
        self.name = name
        self.decoder_type = 'learnable_attention'
        self.finish_setup()

    def forward(self, inputs, attention_mask, encoder_output, encoder_hidden, encoder_cell, decoder_output,
                decoder_hidden, decoder_cell):
        '''
        Pass through one stage of decoding
        :param inputs: Tensor of shape [B] (type long)
        :param attention_mask: Tensor of shape [B, T] (type bool)
        :param encoder_output: Tensor of shape [B, T, H]
        :param encoder_hidden: Tensor of shape [L, B, H]
        :param encoder_cell: Tensor of shape [L, B, H]
        :param decoder_output: Tensor of shape [B, S=1, H]
        :param decoder_hidden: Tensor of shape [L, B, H]
        :param decoder_cell: Tensor of shape [L, B, H]
        :return: log_probs, lstm_output, lstm_hidden, lstm_cell
        log_probs: Tensor of shape [B, V]
        lstm_output: Tensor of shape [B, S=1, H]
        lstm_hidden: Tensor of shape [L, B, H]
        lstm_cell: Tensor of shape [L, B, H]
        '''
        embeds = self.embedding_dropout(self.embedding(inputs.unsqueeze(1)))  # [B, 1, E]
        source_length = encoder_output.shape[1]
        attention_layer_1 = self.attention_layer_1(
            torch.cat([encoder_output, decoder_output.repeat(1, source_length, 1)], dim=2))  # [B, T, H]
        attention_layer_2 = self.attention_layer_2(torch.tanh(attention_layer_1))  # [B, T, 1]
        attention_layer_2 = attention_layer_2.masked_fill(attention_mask.unsqueeze(2), LARGE_NEGATIVE)  # [B, T, 1]
        attention_weights = torch.nn.functional.softmax(attention_layer_2, dim=1).transpose(1, 2)  # [B, 1, T]
        context = torch.bmm(attention_weights, encoder_output)  # [B, 1, H]
        lstm_output, (lstm_hidden, lstm_cell) = self.LSTM(torch.cat([embeds, context], dim=2), (
        decoder_hidden, decoder_cell))  # [B, S=1, H], [L, B, H], [L, B, H]
        proj = self.linear(torch.cat([lstm_output, context, embeds], dim=2).squeeze(1))  # [B, V]
        log_probs = torch.nn.functional.log_softmax(proj, dim=1)  # [B, V]
        return log_probs, lstm_output, lstm_hidden, lstm_cell


class EncoderDecoderLSTM(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers, decoder_type='vanilla', bidirectional_encoder=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 encoder_embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 decoder_embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 encoder_embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 decoder_embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 encoder_inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 decoder_inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 encoder_intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 decoder_intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 use_packing=True, name='LSTM', encoder_name='LSTM', decoder_name='LSTM'):
        super().__init__()
        self.encoder = EncoderLSTM(input_lang, num_layers, bidirectional_encoder, hidden_size,
                                   encoder_embedding_dimension, encoder_embedding_dropout,
                                   encoder_inter_recurrent_layer_dropout, encoder_intra_recurrent_layer_dropout,
                                   use_packing, encoder_name)
        self.decoder_hidden_size = 2 * hidden_size if bidirectional_encoder else hidden_size
        self.decoder_type = decoder_type
        self.output_vocab = output_lang.n_words
        if decoder_type == 'vanilla':
            self.decoder = DecoderLSTM(output_lang, num_layers, self.decoder_hidden_size, decoder_embedding_dimension,
                                       decoder_embedding_dropout, decoder_inter_recurrent_layer_dropout,
                                       decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'context':
            self.decoder = DecoderLSTMWithContext(output_lang, num_layers, self.decoder_hidden_size,
                                                  decoder_embedding_dimension, decoder_embedding_dropout,
                                                  decoder_inter_recurrent_layer_dropout,
                                                  decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'dot_product_attention':
            self.decoder = DecoderLSTMWithDotProductAttention(output_lang, num_layers, self.decoder_hidden_size,
                                                              decoder_embedding_dimension, decoder_embedding_dropout,
                                                              decoder_inter_recurrent_layer_dropout,
                                                              decoder_intra_recurrent_layer_dropout, decoder_name)
        elif decoder_type == 'learnable_attention':
            self.decoder = DecoderLSTMWithLearnableAttention(output_lang, num_layers, self.decoder_hidden_size,
                                                             decoder_embedding_dimension, decoder_embedding_dropout,
                                                             decoder_inter_recurrent_layer_dropout,
                                                             decoder_intra_recurrent_layer_dropout, decoder_name)
        else:
            raise Exception('Decoder type {0} not supported'.format(decoder_type))
        self.name = name + '_' + decoder_type + '_decoder'
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        '''
        Pass through the encoder decoder stack
        :param inputs: Tensor of shape [B, T] (type long)
        :param outputs: Tensor of shape [B, S] (type long)
        :param teacher_force: bool
        :return: all_log_probs
        all_log_probs: Tensor of shape [S, B, V]
        '''
        batch_size, target_length = outputs.shape
        all_log_probs = torch.zeros(target_length, batch_size, self.output_vocab, device=device)  # [S, B, V]
        decoder_input = outputs[:, 0]  # [B]
        encoder_output, encoder_hidden, encoder_cell = self.encoder(inputs)  # [B, T, H], [L, B, H], [L, B, H]
        attention_mask = inputs == data_hyperparameters.PAD_TOKEN  # [B, T]
        decoder_output = encoder_output[:, -1, :].unsqueeze(1)  # [B, 1, H]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        for t in range(1, target_length):
            log_probs, decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, attention_mask,
                                                                                   encoder_output, encoder_hidden,
                                                                                   encoder_cell, decoder_output,
                                                                                   decoder_hidden,
                                                                                   decoder_cell)  # [B, V], [B, 1, H], [L, B, H], [L, B, H]
            all_log_probs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else torch.argmax(log_probs, dim=1)
        return all_log_probs
