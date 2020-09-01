import unicodedata
import re
import torch
import data_hyperparameters
import pickle
from log_utils import create_logger, write_log
from sklearn.model_selection import train_test_split

LOG_FILE = 'data_downloader'
logger = create_logger(LOG_FILE)
device = torch.device('cuda' if data_hyperparameters.USE_CUDA and data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE else 'cpu')
DATA_FILE = 'data/eng-fra.txt'


def save_data(data, path):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()


class Language:
    def __init__(self, name):
        self.name = name
        self.word_to_index = {'SOS': data_hyperparameters.SOS_TOKEN, 'EOS': data_hyperparameters.EOS_TOKEN,
                              'PAD': data_hyperparameters.PAD_TOKEN, 'UNK': data_hyperparameters.UNK_TOKEN}
        self.word_to_count = {}
        self.index_to_word = {data_hyperparameters.SOS_TOKEN: 'SOS', data_hyperparameters.EOS_TOKEN: 'EOS',
                              data_hyperparameters.PAD_TOKEN: 'PAD', data_hyperparameters.UNK_TOKEN: 'UNK'}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.word_to_count[word] = 1
            self.index_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs():
    write_log("Reading lines...", logger)
    # Read the file and split into lines
    lines = open(DATA_FILE, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Language instances
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Language('fr')
    output_lang = Language('en')
    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < data_hyperparameters.MAX_LENGTH and len(
        p[1].split(' ')) < data_hyperparameters.MAX_LENGTH and p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def split_data(input_lang_tokens, output_lang_tokens):
    write_log('Splitting fit data into training and validation sets', logger)
    return train_test_split(input_lang_tokens, output_lang_tokens, test_size=data_hyperparameters.TRAIN_VALID_SPLIT)


def prepare_data():
    input_lang, output_lang, pairs = read_langs()
    write_log("Read {0} sentence pairs".format(len(pairs)), logger)
    pairs = filter_pairs(pairs)
    write_log("Trimmed to {0} sentence pairs".format(len(pairs)), logger)
    write_log("Counting words...", logger)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    write_log("Counted words:", logger)
    write_log('{0}: {1}'.format(input_lang.name, input_lang.n_words), logger)
    write_log('{0}: {1}'.format(output_lang.name, output_lang.n_words), logger)
    return input_lang, output_lang, pairs


def get_langs_and_loaders():
    input_lang, output_lang, pairs = prepare_data()
    input_word_to_index = [
        [data_hyperparameters.SOS_TOKEN] + [input_lang.word_to_index[word] for word in pair[0].split()] + [
            data_hyperparameters.EOS_TOKEN] for pair in pairs]
    output_word_to_index = [
        [data_hyperparameters.SOS_TOKEN] + [output_lang.word_to_index[word] for word in pair[1].split()] + [
            data_hyperparameters.EOS_TOKEN] for pair in pairs]
    input_train, input_valid, output_train, output_valid = split_data(input_word_to_index, output_word_to_index)
    train_data_loader = get_dataloader(input_train, output_train)
    write_log('{0} batches in training data'.format(len(train_data_loader)), logger)
    valid_data_loader = get_dataloader(input_valid, output_valid)
    write_log('{0} batches in validation data'.format(len(valid_data_loader)), logger)
    return input_lang, output_lang, train_data_loader, valid_data_loader


def augment_dataset(input_dataset, output_dataset):
    samples = [(len(input_text), len(output_text), idx, torch.tensor(input_text, device=device),
                torch.tensor(output_text, device=device)) for idx, (input_text, output_text) in
               enumerate(zip(input_dataset, output_dataset))]
    samples.sort()  # sort by length to pad sequences with similar lengths
    return samples


def get_dataloader(input_dataset, output_dataset, batch_size=data_hyperparameters.BATCH_SIZE):
    def pad_batch(batch):
        # Find max length of the batch
        max_input_len = max([sample[0] for sample in batch])
        max_output_len = max([sample[1] for sample in batch])
        xs = [sample[3] for sample in batch]
        ys = [sample[4] for sample in batch]
        xs_padded = torch.stack(
            [torch.cat(
                (x, torch.tensor([data_hyperparameters.PAD_TOKEN] * (max_input_len - len(x)), device=device).long()))
             for x in xs])
        ys_padded = torch.stack(
            [torch.cat(
                (y, torch.tensor([data_hyperparameters.PAD_TOKEN] * (max_output_len - len(y)), device=device).long()))
             for y in ys])
        return xs_padded, ys_padded

    return torch.utils.data.DataLoader(dataset=augment_dataset(input_dataset, output_dataset),
                                       batch_size=batch_size, collate_fn=pad_batch, pin_memory=True)
