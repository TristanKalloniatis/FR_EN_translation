import torch
import data_hyperparameters
import os
from abc import ABC
import matplotlib.pyplot as plt
from datetime import datetime
from math import nan


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
            model_output = model(xb)
            accuracy += model_output.argmax(dim=1).eq(yb).float().mean().item()
            if also_report_model_confidences:
                log_probs, predictions = torch.max(model_output, dim=-1)
                probs = torch.exp(log_probs)
                correct_predictions_mask = torch.where(predictions == yb, torch.ones_like(yb), torch.zeros_like(yb))
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
        self.vocab_size = data_hyperparameters.VOCAB_SIZE
        self.tokenizer = data_hyperparameters.TOKENIZER
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
                      'vocab_size': self.vocab_size, 'tokenizer': self.tokenizer, 'batch_size': self.batch_size}
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


class EncoderGRU(torch.nn.Module):
    def __init__(self, lang, num_layers, hidden_size, dropout, use_packing, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(lang.n_words, hidden_size, padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=hidden_size, hidden_size=hidden_size,
                                batch_first=True, dropout=dropout)
        self.language_name = lang.name
        self.name = name

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


class DecoderGRU(torch.nn.Module):
    def __init__(self, lang, num_layers, hidden_size, dropout, use_packing, use_attention, name):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(lang.n_words, hidden_size, padding_idx=data_hyperparameters.PAD_TOKEN)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.GRU = torch.nn.GRU(num_layers=num_layers, input_size=hidden_size, hidden_size=hidden_size,
                                batch_first=True, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, lang.n_words)
        self.language_name = lang.name
        if use_attention:
            print('Attention not yet implemented')
        self.name = name

    def forward(self, inputs, encoder_output, encoder_hn):
        embeds = self.dropout(self.embedding(inputs))
        if self.use_packing:
            input_length = torch.sum(inputs != data_hyperparameters.PAD_TOKEN, dim=-1)
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, enforce_sorted=False, batch_first=True)
        gru_output, _ = self.GRU(embeds)
        if self.use_packing:
            gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        out = self.linear(gru_output)
        return torch.nn.functional.log_softmax(out, dim=-1)


class EncoderDecoderGRU(torch.nn.Module):
    def __init__(self, input_lang, output_lang, num_layers=1, hidden_size=data_hyperparameters.HIDDEN_SIZE,
                 dropout=data_hyperparameters.DROPOUT, use_packing=True, use_attention=False, name='GRU'):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_packing = use_packing
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.input_language_name = input_lang.name
        self.output_lang_name = output_lang.name
        self.encoder = EncoderGRU(input_lang, num_layers, hidden_size, dropout, use_packing, name)
        self.decoder = DecoderGRU(output_lang, num_layers, hidden_size, dropout, use_packing, use_attention, name)
        self.name = name

    def forward(self, inputs, outputs):
        encoder_output, encoder_hn = self.encoder(inputs)
        return self.decoder(outputs, encoder_output, encoder_hn)
