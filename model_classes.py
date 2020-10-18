import torch
import data_hyperparameters
import os
from abc import ABC
import matplotlib.pyplot as plt
import datetime
from math import nan, log

if not os.path.exists('learning_curves/'):
    os.mkdir('learning_curves/')


def _weight_drop(module, weights, dropout):
    # Replace weight parameters by '_raw' weight parameters
    for weight_name in weights:
        weight = getattr(module, weight_name)
        del module._parameters[weight_name]
        module.register_parameter(weight_name + '_raw', torch.nn.Parameter(weight))
    original_forward = module.forward

    def forward(*args, **kwargs):
        for weight_name in weights:
            weight_raw = getattr(module, weight_name + '_raw')
            weight = torch.nn.Parameter(torch.nn.functional.dropout(weight_raw, p=dropout, training=module.training),
                                        requires_grad=True)
            setattr(module, weight_name, weight)
        return original_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0.0):
        super().__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropGRU(torch.nn.GRU):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = [f'weight_hh_l{i}' for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropLSTM(torch.nn.LSTM):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = [f'weight_hh_l{i}' for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


def get_accuracy(loader, model, also_report_model_confidences=False):
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    model.eval()
    with torch.no_grad():
        accuracy = 0.
        correct_prediction_probs = 0.
        incorrect_prediction_probs = 0.
        num_correct = 0
        num_incorrect = 0
        for xb, yb in loader:
            if data_hyperparameters.USE_CUDA and not data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE:
                xb = xb.cuda()
                yb = yb.cuda()
            model_output = model(xb, yb)
            yb_length = torch.sum(yb != data_hyperparameters.PAD_TOKEN, dim=-1)
            packed_model_output_data = torch.nn.utils.rnn.pack_padded_sequence(model_output, yb_length,
                                                                               batch_first=True,
                                                                               enforce_sorted=False).data
            packed_yb_data = torch.nn.utils.rnn.pack_padded_sequence(yb, yb_length, batch_first=True,
                                                                     enforce_sorted=False).data
            accuracy += packed_model_output_data.argmax(dim=-1).eq(packed_yb_data).float().mean().item()
            if also_report_model_confidences:
                log_probs, predictions = torch.max(packed_model_output_data, dim=-1)
                probs = torch.exp(log_probs)
                correct_predictions_mask = torch.where(predictions == packed_yb_data, torch.ones_like(packed_yb_data),
                                                       torch.zeros_like(packed_yb_data))
                num_correct += torch.sum(correct_predictions_mask).item()
                num_incorrect += torch.sum(1 - correct_predictions_mask).item()
                correct_prediction_probs += torch.sum(correct_predictions_mask * probs).item()
                incorrect_prediction_probs += torch.sum((1 - correct_predictions_mask) * probs).item()
    if also_report_model_confidences:
        return accuracy / len(
            loader), correct_prediction_probs / num_correct, incorrect_prediction_probs / num_incorrect
    else:
        return accuracy / len(loader)


class BaseModelClass(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
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
        self.train_accuracies = {}
        self.valid_accuracies = {}
        self.train_correct_confidences = {}
        self.train_incorrect_confidences = {}
        self.valid_correct_confidences = {}
        self.valid_incorrect_confidences = {}

    def count_parameters(self):
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def finish_setup(self):
        self.count_parameters()

    def get_model_performance_data(self, train_dataloader, valid_dataloader, test_dataloader):
        final_train_loss = nan if len(self.train_losses) == 0 else self.train_losses[-1]
        final_valid_loss = nan if len(self.valid_losses) == 0 else self.valid_losses[-1]
        train_accuracy = get_accuracy(train_dataloader, self)
        valid_accuracy = get_accuracy(valid_dataloader, self)
        test_accuracy = get_accuracy(test_dataloader, self)
        average_time_per_epoch = nan if self.num_epochs_trained == 0 else self.train_time / self.num_epochs_trained
        model_data = {'name': self.name, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                      'test_accuracy': test_accuracy, 'total_train_time': self.train_time,
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

        if len(self.train_accuracies) != 0:
            fig, ax = plt.subplots()
            epochs = list(self.train_accuracies.keys())
            train_accuracies = list(self.train_accuracies.values())
            valid_accuracies = list(self.valid_accuracies.values())
            ax.scatter(epochs, train_accuracies, label='Training')
            ax.scatter(epochs, valid_accuracies, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracies for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/accuracies_{0}.png'.format(self.name))

        if len(self.train_correct_confidences) != 0:
            fig, ax = plt.subplots()
            epochs = list(self.train_correct_confidences.keys())
            train_correct_confidences = list(self.train_correct_confidences.values())
            train_incorrect_confidences = list(self.train_incorrect_confidences.values())
            valid_correct_confidences = list(self.valid_correct_confidences.values())
            valid_incorrect_confidences = list(self.valid_incorrect_confidences.values())
            ax.scatter(epochs, train_correct_confidences, label='Training (correct predictions)')
            ax.scatter(epochs, valid_correct_confidences, label='Validation (correct predictions)')
            ax.scatter(epochs, train_incorrect_confidences, label='Training (incorrect predictions)')
            ax.scatter(epochs, valid_incorrect_confidences, label='Validation (incorrect predictions)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Confidence')
            ax.set_title('Confidences for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/confidences_{0}.png'.format(self.name))

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
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                 bidirectional=bidirectional, input_size=embedding_dimension, hidden_size=hidden_size,
                                 batch_first=True, dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, bidirectional=bidirectional, input_size=embedding_dimension, hidden_size=hidden_size,
            batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        embeds = self.dropout(self.embedding(inputs))
        if self.use_packing:
            input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False,
                                                             batch_first=True)
        gru_output, gru_hn = self.GRU(embeds)
        if self.use_packing:
            gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        return gru_output, gru_hn


class DecoderGRU(BaseModelClass):
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                 input_size=embedding_dimension,
                                 hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True,
                                 dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension,
            hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn):
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        gru_output, gru_hn = self.GRU(embeds, encoder_input)
        return torch.nn.functional.log_softmax(self.linear(gru_output.squeeze()), dim=-1), gru_output, gru_hn


class DecoderGRUWithContext(BaseModelClass): # todo: make this work with multiple layers
    def __init__(self, lang, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout, num_layers=1,
                                 input_size=embedding_dimension + 2 * hidden_size if bidirectional else embedding_dimension + hidden_size,
                                 hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(4 * hidden_size + embedding_dimension if bidirectional else 2 * hidden_size + embedding_dimension,
                                      lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn):
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        embeds_with_context = torch.cat([embeds, encoder_input.transpose(0, 1)], dim=-1)
        gru_output, gru_hn = self.GRU(embeds_with_context, encoder_input)
        output_with_embeds_and_context = torch.cat([embeds_with_context.squeeze(1), gru_output.squeeze(1)], dim=-1)
        return torch.nn.functional.log_softmax(self.linear(output_with_embeds_and_context), dim=-1), gru_output, gru_hn


class DecoderGRUWithDotProductAttention(BaseModelClass): # todo: make work when num_layers > 1
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                 input_size=embedding_dimension,
                                 hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True,
                                 dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension,
            hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn):
        pass
        # todo: finish this
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        similarity_scores = torch.bmm(encoder_output, )
        embeds_with_context = torch.cat([embeds, encoder_input.transpose(0, 1)], dim=-1)
        gru_output, gru_hn = self.GRU(embeds_with_context, encoder_input)
        output_with_embeds_and_context = torch.cat([embeds_with_context.squeeze(1), gru_output.squeeze(1)], dim=-1)
        return torch.nn.functional.log_softmax(self.linear(output_with_embeds_and_context), dim=-1), gru_output, gru_hn


class DecoderGRUWithLearnableAttention(BaseModelClass): # todo: make work when num_layers > 1
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                 input_size=embedding_dimension,
                                 hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True,
                                 dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropGRU(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension,
            hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn):
        pass
        # todo: finish this
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        similarity_scores = torch.bmm(encoder_output, )
        embeds_with_context = torch.cat([embeds, encoder_input.transpose(0, 1)], dim=-1)
        gru_output, gru_hn = self.GRU(embeds_with_context, encoder_input)
        output_with_embeds_and_context = torch.cat([embeds_with_context.squeeze(1), gru_output.squeeze(1)], dim=-1)
        return torch.nn.functional.log_softmax(self.linear(output_with_embeds_and_context), dim=-1), gru_output, gru_hn


class EncoderDecoderGRU(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 use_packing=True, name='GRU'):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                  embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                  use_packing, name)
        self.decoder = DecoderGRU(output_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                  embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn = self.decoder(decoder_input, encoder_output, encoder_hn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderGRUWithContext(BaseModelClass):
    def __init__(self, input_lang, output_lang, bidirectional=False, hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='GRUWithContext'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, 1, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                                  inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, use_packing, name)
        self.decoder = DecoderGRUWithContext(output_lang, bidirectional, hidden_size, embedding_dimension,
                                             embedding_dropout, inter_recurrent_layer_dropout,
                                             intra_recurrent_layer_dropout, name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn = self.decoder(decoder_input, encoder_output, encoder_hn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderGRUWithDotProductAttention(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='GRUWithDotProductAttention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                  embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                  use_packing, name)
        self.decoder = DecoderGRUWithDotProductAttention(output_lang, num_layers, bidirectional, hidden_size,
                                                         embedding_dimension, embedding_dropout,
                                                         inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                                         name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn = self.decoder(decoder_input, encoder_output, encoder_hn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderGRUWithLearnableAttention(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='GRUWithLearnableAttention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                  embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                  use_packing, name)
        self.decoder = DecoderGRUWithDotProductAttention(output_lang, num_layers, bidirectional, hidden_size,
                                                         embedding_dimension, embedding_dropout,
                                                         inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                                         name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn = self.decoder(decoder_input, encoder_output, encoder_hn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderLSTM(BaseModelClass):
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, use_packing, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                   input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True,
                                   dropout=inter_recurrent_layer_dropout, bidirectional=bidirectional) if num_layers > 1 else WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.language_name = lang.name
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        embeds = self.dropout(self.embedding(inputs))
        if self.use_packing:
            input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False,
                                                             batch_first=True)
        lstm_output, (lstm_hn, lstm_cn) = self.LSTM(embeds)
        if self.use_packing:
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        return lstm_output, lstm_hn, lstm_cn


class DecoderLSTM(BaseModelClass):
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                   input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size,
                                   batch_first=True, dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn, encoder_cn):
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input_hn = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        encoder_input_cn = encoder_cn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_cn
        lstm_output, (lstm_hn, lstm_cn) = self.LSTM(embeds, (encoder_input_hn, encoder_input_cn))
        return torch.nn.functional.log_softmax(self.linear(lstm_output.squeeze()),
                                               dim=-1), lstm_output, lstm_hn, lstm_cn


class DecoderLSTMWithContext(BaseModelClass): # todo: make work with multiple layers
    def __init__(self, lang, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout, num_layers=1,
                                   input_size=embedding_dimension + 2 * hidden_size if bidirectional else embedding_dimension + hidden_size,
                                   hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(4 * hidden_size + embedding_dimension if bidirectional else 2 * hidden_size + embedding_dimension,
                                      lang.n_words)
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn, encoder_cn):
        embeds = self.dropout(self.embedding(inputs.unsqueeze(1)))
        encoder_input_hn = encoder_hn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_hn
        encoder_input_cn = encoder_cn.view(self.num_layers, 2, -1, self.hidden_size).transpose(1, 2).reshape(self.num_layers, -1, 2 * self.hidden_size) if self.bidirectional else encoder_cn
        embeds_with_context = torch.cat([embeds, encoder_input_hn.transpose(0, 1)], dim=-1)
        lstm_output, (lstm_hn, lstm_cn) = self.LSTM(embeds_with_context, (encoder_input_hn, encoder_input_cn))
        output_with_embeds_and_context = torch.cat([embeds_with_context.squeeze(1), lstm_output.squeeze(1)], dim=-1)
        return torch.nn.functional.log_softmax(self.linear(output_with_embeds_and_context),
                                               dim=-1), lstm_output, lstm_hn, lstm_cn


class DecoderLSTMWithDotProductAttention(BaseModelClass): # todo: make work when num_layers > 1
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                   input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size,
                                   batch_first=True, dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self):
        # todo: finish this
        pass


class DecoderLSTMWithLearnableAttention(BaseModelClass): # todo: make work when num_layers > 1
    def __init__(self, lang, num_layers, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                 inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dimension = embedding_dimension
        self.vocab_size = lang.n_words
        self.embedding = torch.nn.Embedding(self.vocab_size, embedding_dimension,
                                            padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout, num_layers=num_layers,
                                   input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size,
                                   batch_first=True, dropout=inter_recurrent_layer_dropout) if num_layers > 1 else WeightDropLSTM(weight_dropout=intra_recurrent_layer_dropout,
            num_layers=num_layers, input_size=embedding_dimension, hidden_size=2 * hidden_size if bidirectional else hidden_size, batch_first=True)
        self.language_name = lang.name
        self.name = name
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, lang.n_words)
        self.finish_setup()

    def forward(self):
        # todo: finish this
        pass


class EncoderDecoderLSTM(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT,
                 use_packing=True, name='LSTM'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderLSTM(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                   embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                   use_packing, name)
        self.decoder = DecoderLSTM(output_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                   embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                   name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn, encoder_cn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn, decoder_cn = self.decoder(decoder_input, encoder_output, encoder_hn,
                                                                             encoder_cn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderLSTMWithContext(BaseModelClass):
    def __init__(self, input_lang, output_lang, bidirectional=False, hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='LSTMWithContext'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderLSTM(input_lang, 1, bidirectional, hidden_size, embedding_dimension, embedding_dropout,
                                   inter_recurrent_layer_dropout, intra_recurrent_layer_dropout, use_packing, name)
        self.decoder = DecoderLSTMWithContext(output_lang, bidirectional, hidden_size, embedding_dimension,
                                              embedding_dropout, inter_recurrent_layer_dropout,
                                              intra_recurrent_layer_dropout, name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn, encoder_cn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn, decoder_cn = self.decoder(decoder_input, encoder_output, encoder_hn,
                                                                             encoder_cn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderLSTMWithDotProductAttention(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='LSTMWithDotProductAttention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderLSTM(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                   embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                   use_packing, name)
        self.decoder = DecoderLSTMWithDotProductAttention(output_lang, num_layers, bidirectional, hidden_size,
                                                          embedding_dimension, embedding_dropout,
                                                          inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                                          name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn, encoder_cn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn, decoder_cn = self.decoder(decoder_input, encoder_output, encoder_hn,
                                                                             encoder_cn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs


class EncoderDecoderLSTMWithLearnableAttention(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, bidirectional=False,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 inter_recurrent_layer_dropout=data_hyperparameters.INTER_RECURRENT_LAYER_DROPOUT,
                 intra_recurrent_layer_dropout=data_hyperparameters.INTRA_RECURRENT_LAYER_DROPOUT, use_packing=True,
                 name='LSTMWithLearnableAttention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderLSTM(input_lang, num_layers, bidirectional, hidden_size, embedding_dimension,
                                   embedding_dropout, inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                   use_packing, name)
        self.decoder = DecoderLSTMWithDotProductAttention(output_lang, num_layers, bidirectional, hidden_size,
                                                          embedding_dimension, embedding_dropout,
                                                          inter_recurrent_layer_dropout, intra_recurrent_layer_dropout,
                                                          name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, teacher_force=False):
        batch_size = outputs.shape[0]
        output_length = outputs.shape[1]
        encoder_output, encoder_hn, encoder_cn = self.encoder(inputs)
        decoder_input = outputs[:, 0]
        decoder_outputs = torch.zeros(output_length, batch_size, self.decoder.vocab_size,
                                      device='cuda' if data_hyperparameters.USE_CUDA else 'cpu')
        for t in range(1, output_length):
            log_probs, decoder_output, decoder_hn, decoder_cn = self.decoder(decoder_input, encoder_output, encoder_hn,
                                                                             encoder_cn)
            decoder_outputs[t] = log_probs
            decoder_input = outputs[:, t] if teacher_force else log_probs.argmax(dim=-1)
        return decoder_outputs
