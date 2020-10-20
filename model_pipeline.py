import torch
import data_hyperparameters
from datetime import datetime
import os
import csv
from log_utils import create_logger, write_log
from pickle import dump, load
from data_downloader import normalize_string
from random import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

LOG_FILE = 'model_pipeline'
logger = create_logger(LOG_FILE)

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models/')

device = torch.device('cuda' if data_hyperparameters.USE_CUDA else 'cpu')


def train(model, train_data, valid_data, epochs=data_hyperparameters.EPOCHS, patience=data_hyperparameters.PATIENCE,
          teacher_forcing_scale_factor=data_hyperparameters.TEACHER_FORCING_SCALE_FACTOR):
    loss_function = torch.nn.NLLLoss(ignore_index=data_hyperparameters.PAD_TOKEN)
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    optimiser = torch.optim.Adam(model.parameters()) if model.latest_scheduled_lr is None else torch.optim.Adam(
        model.parameters(), lr=model.latest_scheduled_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=patience, mode='max')
    now_begin_training = datetime.now()
    start_epoch = model.num_epochs_trained
    for epoch in range(start_epoch, epochs + start_epoch):
        now_begin_epoch = datetime.now()
        model.latest_scheduled_lr = optimiser.param_groups[0]['lr']
        model.teacher_forcing_proportion_history.append(model.teacher_forcing_proportion)
        model.lr_history.append(model.latest_scheduled_lr)
        write_log('Running epoch {0} of {1} with learning rate {2} and teacher forcing rate {3}'.format(epoch + 1,
                                                                                                        epochs + start_epoch,
                                                                                                        model.latest_scheduled_lr,
                                                                                                        model.teacher_forcing_proportion),
                  logger)
        model.train()
        loss = 0.
        for xb, yb in train_data:
            if data_hyperparameters.USE_CUDA and not data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE:
                xb = xb.cuda()
                yb = yb.cuda()
            teacher_force = random() < model.teacher_forcing_proportion
            batch_loss = loss_function(
                torch.flatten(model(xb, yb, teacher_force=teacher_force)[1:], start_dim=0, end_dim=1),
                torch.flatten(yb.transpose(0, 1)[1:]))
            loss += batch_loss.item() / len(train_data)
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()
        model.train_losses.append(loss)
        write_log('Training loss: {0}'.format(loss), logger)
        model.eval()
        train_bleu = average_bleu(train_data, model)
        write_log('Training BLEU: {0}'.format(train_bleu), logger)
        model.train_bleus.append(train_bleu)
        with torch.no_grad():
            if data_hyperparameters.USE_CUDA and not data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE:
                loss = 0.
                for xb, yb in valid_data:
                    xb = xb.cuda()
                    yb = yb.cuda()
                    loss += loss_function(torch.flatten(model(xb, yb, teacher_force=False)[1:], start_dim=0, end_dim=1),
                                          torch.flatten(yb.transpose(0, 1)[1:])).item() / len(valid_data)
            else:
                loss = sum([loss_function(torch.flatten(model(xb, yb, teacher_force=False)[1:], start_dim=0, end_dim=1),
                                          torch.flatten(yb.transpose(0, 1)[1:])).item() for xb, yb in
                            valid_data]) / len(valid_data)
        model.valid_losses.append(loss)
        write_log('Validation loss: {0}'.format(loss), logger)
        valid_bleu = average_bleu(valid_data, model)
        write_log('Validation BLEU: {0}'.format(valid_bleu), logger)
        model.valid_bleus.append(valid_bleu)
        scheduler.step(valid_bleu)
        model.num_epochs_trained += 1
        model.teacher_forcing_proportion *= teacher_forcing_scale_factor
        write_log('Epoch took {0} seconds'.format((datetime.now() - now_begin_epoch).total_seconds()), logger)
    model.train_time += (datetime.now() - now_begin_training).total_seconds()
    if data_hyperparameters.USE_CUDA:
        model.cpu()


def report_statistics(model, train_data, valid_data, test_data):
    save_model(model)
    model_data = model.get_model_performance_data(train_data, valid_data, test_data)
    if not os.path.isfile(data_hyperparameters.STATISTICS_FILE):
        with open(data_hyperparameters.STATISTICS_FILE, 'w') as f:
            w = csv.DictWriter(f, model_data.keys())
            w.writeheader()
            w.writerow(model_data)
    else:
        with open(data_hyperparameters.STATISTICS_FILE, 'a') as f:
            w = csv.DictWriter(f, model_data.keys())
            w.writerow(model_data)


def save_model(model):
    torch.save(model.state_dict(), 'saved_models/{0}.pt'.format(model.name))
    model_data = {'train_losses': model.train_losses, 'valid_losses': model.valid_losses,
                  'train_bleus': model.train_bleus, 'valid_bleus': model.valid_bleus,
                  'num_epochs_trained': model.num_epochs_trained, 'latest_scheduled_lr': model.latest_scheduled_lr,
                  'lr_history': model.lr_history, 'train_time': model.train_time,
                  'num_trainable_params': model.num_trainable_params, 'instantiated': model.instantiated,
                  'name': model.name, 'batch_size': model.batch_size,
                  'teacher_forcing_proportion': model.teacher_forcing_proportion,
                  'teacher_forcing_proportion_history': model.teacher_forcing_proportion_history}
    outfile = open('saved_models/{0}_model_data.pkl'.format(model.name), 'wb')
    dump(model_data, outfile)
    outfile.close()


def load_model_state(model, model_name):
    model.load_state_dict(torch.load('saved_models/{0}.pt'.format(model_name)))
    write_log('Loaded model {0} weights'.format(model_name), logger)
    infile = open('saved_models/{0}_model_data.pkl'.format(model_name), 'rb')
    model_data = load(infile)
    infile.close()
    model.train_losses = model_data['train_losses']
    model.valid_losses = model_data['valid_losses']
    model.train_bleus = model_data['train_bleus']
    model.valid_bleus = model_data['valid_bleus']
    model.num_epochs_trained = model_data['num_epochs_trained']
    model.latest_scheduled_lr = model_data['latest_scheduled_lr']
    model.lr_history = model_data['lr_history']
    model.train_time = model_data['train_time']
    model.num_trainable_params = model_data['num_trainable_params']
    model.instantiated = model_data['instantiated']
    model.name = model_data['name']
    model.batch_size = model_data['batch_size']
    model.teacher_forcing_proportion = model_data['teacher_forcing_proportion']
    model.teacher_forcing_proportion_history = model_data['teacher_forcing_proportion_history']
    write_log('Loaded model {0} state'.format(model_name), logger)


def remove_tokens(indices):
    result = []
    for i in range(1, len(indices)):
        if indices[i] != data_hyperparameters.EOS_TOKEN:
            result.append(indices[i])
        else:
            break
    return result


def evaluate_bleu_index_batch(reference_batch, candidate_batch):
    batch_size = reference_batch.shape[0]
    chencherry = SmoothingFunction()
    results = torch.zeros(batch_size, device=device)
    for b in range(batch_size):
        reference = remove_tokens(reference_batch[b, :].tolist())
        candidate = remove_tokens(candidate_batch[b, :].tolist())
        results[b] = sentence_bleu([reference], candidate, smoothing_function=chencherry.method1)
    return results


def average_bleu(data, model):
    avg_bleu = 0.
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    with torch.no_grad():
        model.eval()
        for xb, yb in data:
            if data_hyperparameters.USE_CUDA and not data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE:
                xb = xb.cuda()
                yb = yb.cuda()
            translations = torch.argmax(model(xb, yb, teacher_force=False).transpose(0, 1), dim=2)  # [B, S]
            bleu_batch = evaluate_bleu_index_batch(yb, translations)
            avg_bleu += torch.mean(bleu_batch).item()
    return 100 * avg_bleu / len(data)


def translate(sentence, input_language, output_language, model):
    tokenized_sentence = [t.text for t in input_language.tokenizer(normalize_string(sentence.strip()))]
    input_index = torch.tensor(input_language.index_sentences([tokenized_sentence]), device=device)
    output_index = torch.zeros(1, data_hyperparameters.MAX_LENGTH, dtype=torch.long, device=device).fill_(
        data_hyperparameters.SOS_TOKEN)
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    with torch.no_grad():
        model.eval()
        output_index = torch.argmax(model(input_index, output_index, teacher_force=False).transpose(0, 1),
                                    dim=2).squeeze(0).tolist()
    translation = ''
    for index in output_index:
        if index == data_hyperparameters.EOS_TOKEN:
            break
        translation = translation + ' ' + output_language.index_to_word[index]
    return translation.strip()
