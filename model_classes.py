import torch
import data_hyperparameters
import os
from abc import ABC
import matplotlib.pyplot as plt
import datetime
from math import nan, log


if not os.path.exists('learning_curves/'):
    os.mkdir('learning_curves/')


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
            packed_model_output_data = torch.nn.utils.rnn.pack_padded_sequence(model_output, yb_length, batch_first=True, enforce_sorted=False).data
            packed_yb_data = torch.nn.utils.rnn.pack_padded_sequence(yb, yb_length, batch_first=True, enforce_sorted=False).data
            accuracy += packed_model_output_data.argmax(dim=-1).eq(packed_yb_data).float().mean().item()
            if also_report_model_confidences:
                log_probs, predictions = torch.max(packed_model_output_data, dim=-1)
                probs = torch.exp(log_probs)
                correct_predictions_mask = torch.where(predictions == packed_yb_data, torch.ones_like(packed_yb_data), torch.zeros_like(packed_yb_data))
                num_correct += torch.sum(correct_predictions_mask).item()
                num_incorrect += torch.sum(1 - correct_predictions_mask).item()
                correct_prediction_probs += torch.sum(correct_predictions_mask * probs).item()
                incorrect_prediction_probs += torch.sum((1 - correct_predictions_mask) * probs).item()
    if also_report_model_confidences:
        return accuracy / len(loader), correct_prediction_probs / num_correct, incorrect_prediction_probs / num_incorrect
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
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, dropout, use_packing, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(lang.n_words, embedding_dimension, padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True, dropout=dropout) if num_layers > 1 else torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True)
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
    def __init__(self, lang, num_layers, hidden_size, embedding_dimension, dropout, use_packing, use_attention, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(lang.n_words, embedding_dimension, padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True, dropout=dropout) if num_layers > 1 else torch.nn.GRU(num_layers=num_layers, input_size=embedding_dimension, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, lang.n_words)
        self.language_name = lang.name
        if use_attention:
            print('Attention not yet implemented')
        self.name = name
        self.finish_setup()

    def forward(self, inputs, encoder_output, encoder_hn, mode='forcing', return_sequences=False):
        if mode.lower() == 'forcing':
            embeds = self.dropout(self.embedding(inputs))
            if self.use_packing:
                input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)
                embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False, batch_first=True)
            gru_output, _ = self.GRU(embeds, encoder_hn)
            if self.use_packing:
                gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
            out = self.linear(gru_output)
            return torch.nn.functional.log_softmax(out, dim=-1)
        else:
            batch_size = encoder_output.shape[0]
            translation = [data_hyperparameters.SOS_TOKEN] * batch_size
            translation = torch.tensor(translation).unsqueeze(-1)
            max_log_probs = torch.zeros(batch_size, 1)
            if data_hyperparameters.USE_CUDA:
                translation = translation.cuda()
                max_log_probs = max_log_probs.cuda()
            for _ in range(data_hyperparameters.MAX_LENGTH):
                embeds = self.dropout(self.embedding(translation))
                if self.use_packing:
                    translation_length = torch.sum(translation != data_hyperparameters.PAD_TOKEN, dim=-1)
                    embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, translation_length, enforce_sorted=False,
                                                                     batch_first=True)
                gru_output, _ = self.GRU(embeds, encoder_hn)
                if self.use_packing:
                    gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
                out = self.linear(gru_output)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1)
                next_max_log_probs, next_indices = torch.max(log_probs, dim=-1)
                next_max_log_probs = next_max_log_probs[:, -1].unsqueeze(-1)
                next_indices = next_indices[:, -1].unsqueeze(-1)
                translation = torch.cat([translation, next_indices], dim=-1)
                max_log_probs = torch.cat([max_log_probs, next_max_log_probs], dim=-1)
            return max_log_probs, translation if return_sequences else max_log_probs



class EncoderDecoderGRU(BaseModelClass):
    def __init__(self, input_lang, output_lang, num_layers=1, hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION, dropout=data_hyperparameters.DROPOUT,
                 use_packing=True, use_attention=False, name='GRU'):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, num_layers, hidden_size, embedding_dimension, dropout, use_packing, name)
        self.decoder = DecoderGRU(output_lang, num_layers, hidden_size, embedding_dimension, dropout, use_packing,
                                  use_attention, name)
        self.name = name
        self.finish_setup()

    def forward(self, inputs, outputs, mode='forcing', return_sequences=False):
        encoder_output, encoder_hn = self.encoder(inputs)
        return self.decoder(outputs, encoder_output, encoder_hn, mode, return_sequences)


class PositionalEncoding(torch.nn.Module):
    # Modified from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        assert d_model % 2 == 0
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [T, B, d_model] -> [T, B, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(BaseModelClass, ABC):
    def __init__(self, input_lang, output_lang, num_layers=1, max_len=500,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 nhead=data_hyperparameters.ATTENTION_HEADS, dim_feedforward=data_hyperparameters.FEEDFORWARD_DIMENSION,
                 dropout=data_hyperparameters.DROPOUT,
                 positional_encoding_dropout=data_hyperparameters.POSITIONAL_ENCODER_DROPOUT,
                 name='TransformerEncoder'):
        super().__init__()
        assert embedding_dimension % nhead == 0
        self.max_len = max_len
        self.name = name
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.embedding_dimension = embedding_dimension
        self.input_embedding = torch.nn.Embedding(input_lang.n_words, embedding_dimension,
                                                  padding_idx=data_hyperparameters.PAD_TOKEN)
        self.output_embedding = torch.nn.Embedding(output_lang.n_words, embedding_dimension,
                                                   padding_idx=data_hyperparameters.PAD_TOKEN)
        self.input_positional_encoder = PositionalEncoding(embedding_dimension, max_len=max_len,
                                                           dropout=positional_encoding_dropout)
        self.output_positional_encoder = PositionalEncoding(embedding_dimension, max_len=max_len,
                                                            dropout=positional_encoding_dropout)
        self.transformer = torch.nn.Transformer(embedding_dimension, nhead, num_layers, num_layers, dim_feedforward, dropout)
        self.linear = torch.nn.Linear(embedding_dimension, output_lang.n_words)
        self.finish_setup()

    def forward(self, inputs, outputs, mode='forcing', return_sequences=False):
        if mode.lower() == 'forcing':
            input_truncated = inputs[:, :self.max_len]
            output_truncated = outputs[:, :self.max_len]
            input_embeds = self.input_embedding(input_truncated).transpose(0, 1)
            output_embeds = self.output_embedding(output_truncated).transpose(0, 1)
            input_positional_encodings = self.input_positional_encoder(input_embeds)
            output_positional_encodings = self.output_positional_encoder(output_embeds)
            src_key_padding_mask = (input_truncated == data_hyperparameters.PAD_TOKEN)
            tgt_key_padding_mask = (output_truncated == data_hyperparameters.PAD_TOKEN)
            transforms = self.transformer(input_positional_encodings, output_positional_encodings,
                                          src_key_padding_mask=src_key_padding_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)
            out = self.linear(transforms.transpose(0, 1))
            return torch.nn.functional.log_softmax(out, dim=-1)
        else:
            batch_size = inputs.shape[0]
            translation = [data_hyperparameters.SOS_TOKEN] * batch_size
            translation = torch.tensor(translation).unsqueeze(-1)
            max_log_probs = torch.zeros(batch_size, 1)
            if data_hyperparameters.USE_CUDA:
                translation = translation.cuda()
                max_log_probs = max_log_probs.cuda()
            for _ in range(data_hyperparameters.MAX_LENGTH):
                input_embeds = self.input_embedding(inputs).transpose(0, 1)
                output_embeds = self.output_embedding(translation).transpose(0, 1)
                input_positional_encodings = self.input_positional_encoder(input_embeds)
                output_positional_encodings = self.output_positional_encoder(output_embeds)
                src_key_padding_mask = (inputs == data_hyperparameters.PAD_TOKEN)
                tgt_key_padding_mask = (translation == data_hyperparameters.PAD_TOKEN)
                transforms = self.transformer(input_positional_encodings, output_positional_encodings,
                                              src_key_padding_mask=src_key_padding_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask)
                out = self.linear(transforms.transpose(0, 1))
                log_probs = torch.nn.functional.log_softmax(out, dim=-1)
                next_max_log_probs, next_indices = torch.max(log_probs, dim=-1)
                next_max_log_probs = next_max_log_probs[:, -1].unsqueeze(-1)
                next_indices = next_indices[:, -1].unsqueeze(-1)
                translation = torch.cat([translation, next_indices], dim=-1)
                max_log_probs = torch.cat([max_log_probs, next_max_log_probs], dim=-1)
            return max_log_probs, translation if return_sequences else max_log_probs


class PackedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.NLLLoss()

    def forward(self, pred_probs, actuals):
        actuals_length = torch.sum(actuals != data_hyperparameters.PAD_TOKEN, dim=-1)
        packed_pred_probs_data = torch.nn.utils.rnn.pack_padded_sequence(pred_probs, actuals_length, batch_first=True, enforce_sorted=False).data
        packed_actuals_data = torch.nn.utils.rnn.pack_padded_sequence(actuals, actuals_length, batch_first=True, enforce_sorted=False).data
        return self.loss_function(packed_pred_probs_data, packed_actuals_data)
