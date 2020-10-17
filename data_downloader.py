import unicodedata
import re
import torch
import spacy
import data_hyperparameters
from log_utils import create_logger, write_log
from sklearn.model_selection import train_test_split
import pickle
import os

LOG_FILE = 'data_downloader'
logger = create_logger(LOG_FILE)
device = torch.device('cuda' if data_hyperparameters.USE_CUDA and data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE else 'cpu')
DATA_FILE = 'data/eng-fra.txt'
EN_WORD_TO_INDEX_FILE = 'english_WORD_TO_INDEX.pkl'
EN_WORD_TO_COUNT_FILE = 'english_WORD_TO_COUNT.pkl'
EN_INDEX_TO_WORD_FILE = 'english_INDEX_TO_WORD.pkl'
FR_WORD_TO_INDEX_FILE = 'french_WORD_TO_INDEX.pkl'
FR_WORD_TO_COUNT_FILE = 'french_WORD_TO_COUNT.pkl'
FR_INDEX_TO_WORD_FILE = 'french_INDEX_TO_WORD.pkl'
EN_FIT_INDEX_FILE = 'english_FIT_INDEX.pkl'
EN_VALID_INDEX_FILE = 'english_VALID_INDEX.pkl'
EN_TEST_INDEX_FILE = 'english_TEST_INDEX.pkl'
FR_FIT_INDEX_FILE = 'french_FIT_INDEX.pkl'
FR_VALID_INDEX_FILE = 'french_VALID_INDEX.pkl'
FR_TEST_INDEX_FILE = 'french_TEST_INDEX.pkl'


def save_data(data, path):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()


def prepare_data():
    english = Language('english')
    french = Language('french')
    if not os.path.exists(EN_WORD_TO_INDEX_FILE) or not os.path.exists(EN_WORD_TO_COUNT_FILE) or not os.path.exists(EN_INDEX_TO_WORD_FILE) or not os.path.exists(FR_WORD_TO_INDEX_FILE) or not os.path.exists(FR_WORD_TO_COUNT_FILE) or not os.path.exists(FR_INDEX_TO_WORD_FILE) or not os.path.exists(EN_FIT_INDEX_FILE) or not os.path.exists(EN_VALID_INDEX_FILE) or not os.path.exists(EN_TEST_INDEX_FILE) or not os.path.exists(FR_FIT_INDEX_FILE) or not os.path.exists(FR_VALID_INDEX_FILE) or not os.path.exists(FR_TEST_INDEX_FILE):
        en_sentences = []
        fr_sentences = []
        en_tokenizer = spacy.load('en').tokenizer
        fr_tokenizer = spacy.load('fr').tokenizer
        write_log('Reading sentence pairs from file', logger)
        with open(DATA_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                en_sentence, fr_sentence = line.strip().split('\t')
                en_sentence = [t.text for t in en_tokenizer(normalize_string(en_sentence))]
                fr_sentence = [t.text for t in fr_tokenizer(normalize_string(fr_sentence))]
                #todo: remove this filtration
                if len(en_sentence) > data_hyperparameters.MAX_LENGTH or len(fr_sentence) > data_hyperparameters.MAX_LENGTH:
                    continue
                en_sentences.append(en_sentence)
                fr_sentences.append(fr_sentence)
        write_log('Splitting data', logger)
        en_sentences_train, en_sentences_test, fr_sentences_train, fr_sentences_test = train_test_split(en_sentences,
                                                                                                        fr_sentences,
                                                                                                        test_size=data_hyperparameters.TRAIN_TEST_SPLIT)
        en_sentences_fit, en_sentences_valid, fr_sentences_fit, fr_sentences_valid = train_test_split(en_sentences_train,
                                                                                                      fr_sentences_train,
                                                                                                      test_size=data_hyperparameters.TRAIN_VALID_SPLIT)
        write_log('Building languages', logger)
        english.read(en_sentences_fit)
        english.cache()
        french.read(fr_sentences_fit)
        french.cache()
        write_log('Indexing sentences', logger)
        en_sentences_fit_index = english.index_sentences(en_sentences_fit)
        save_data(en_sentences_fit_index, EN_FIT_INDEX_FILE)
        en_sentences_valid_index = english.index_sentences(en_sentences_valid)
        save_data(en_sentences_valid_index, EN_VALID_INDEX_FILE)
        en_sentences_test_index = english.index_sentences(en_sentences_test)
        save_data(en_sentences_test_index, EN_TEST_INDEX_FILE)
        fr_sentences_fit_index = french.index_sentences(fr_sentences_fit)
        save_data(fr_sentences_fit_index, FR_FIT_INDEX_FILE)
        fr_sentences_valid_index = french.index_sentences(fr_sentences_valid)
        save_data(fr_sentences_valid_index, FR_VALID_INDEX_FILE)
        fr_sentences_test_index = french.index_sentences(fr_sentences_test)
        save_data(fr_sentences_test_index, FR_TEST_INDEX_FILE)
    else:
        write_log('Loading languages from disk', logger)
        english.load()
        french.load()
        write_log('Loading indexed sentences from disk', logger)
        en_sentences_fit_index = pickle.load(open(EN_FIT_INDEX_FILE, 'rb'))
        en_sentences_valid_index = pickle.load(open(EN_VALID_INDEX_FILE, 'rb'))
        en_sentences_test_index = pickle.load(open(EN_TEST_INDEX_FILE, 'rb'))
        fr_sentences_fit_index = pickle.load(open(FR_FIT_INDEX_FILE, 'rb'))
        fr_sentences_valid_index = pickle.load(open(FR_VALID_INDEX_FILE, 'rb'))
        fr_sentences_test_index = pickle.load(open(FR_TEST_INDEX_FILE, 'rb'))
    train_data_loader = get_dataloader(fr_sentences_fit_index, en_sentences_fit_index)
    write_log('{0} batches in training data'.format(len(train_data_loader)), logger)
    valid_data_loader = get_dataloader(fr_sentences_valid_index, en_sentences_valid_index)
    write_log('{0} batches in validation data'.format(len(valid_data_loader)), logger)
    test_data_loader = get_dataloader(fr_sentences_test_index, en_sentences_test_index)
    write_log('{0} batches in test data'.format(len(test_data_loader)), logger)
    return french, english, train_data_loader, valid_data_loader, test_data_loader


class Language:
    def __init__(self, name, min_occurences=data_hyperparameters.MIN_OCCURENCES):
        self.name = name
        self.word_to_index = {'<SOS>': data_hyperparameters.SOS_TOKEN, '<EOS>': data_hyperparameters.EOS_TOKEN,
                              '<PAD>': data_hyperparameters.PAD_TOKEN, '<UNK>': data_hyperparameters.UNK_TOKEN}
        self.word_to_count = {}
        self.index_to_word = {data_hyperparameters.SOS_TOKEN: '<SOS>', data_hyperparameters.EOS_TOKEN: '<EOS>',
                              data_hyperparameters.PAD_TOKEN: '<PAD>', data_hyperparameters.UNK_TOKEN: '<UNK>'}
        self.n_words = 4
        self.min_occurences = min_occurences

    def read(self, sentences):
        for sentence in sentences:
            for token in sentence:
                if token in self.word_to_count:
                    self.word_to_count[token] += 1
                else:
                    self.word_to_count[token] = 1
        for token in self.word_to_count:
            if self.word_to_count[token] >= self.min_occurences:
                self.word_to_index[token] = self.n_words
                self.index_to_word[self.n_words] = token
                self.n_words += 1
        for token in list(self.word_to_count):
            if self.word_to_count[token] < self.min_occurences:
                del self.word_to_count[token]

    def cache(self):
        save_data(self.word_to_index, self.name + '_WORD_TO_INDEX.pkl')
        save_data(self.word_to_count, self.name + '_WORD_TO_COUNT.pkl')
        save_data(self.index_to_word, self.name + '_INDEX_TO_WORD.pkl')

    def load(self):
        self.word_to_index = pickle.load(open(self.name + '_WORD_TO_INDEX.pkl', 'rb'))
        self.word_to_count = pickle.load(open(self.name + '_WORD_TO_COUNT.pkl', 'rb'))
        self.index_to_word = pickle.load(open(self.name + '_INDEX_TO_WORD.pkl', 'rb'))
        self.n_words = len(self.word_to_index)

    def index_sentences(self, sentences):
        return [[(self.word_to_index[token] if token in self.word_to_index else self.word_to_index['<UNK>']) for token in ['<SOS>'] + sentence + ['<EOS>']] for sentence in sentences]


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return ''.join(c for c in s if c.isalpha() or c == ' ')


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
