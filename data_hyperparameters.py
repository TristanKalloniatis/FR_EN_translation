from torch.cuda import is_available

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3
MAX_LENGTH = 10
BATCH_SIZE = 32
USE_CUDA = is_available()
STORE_DATA_ON_GPU_IF_AVAILABLE = False
TRAIN_VALID_SPLIT = 0.1
HIDDEN_SIZE = 64
DROPOUT = 0.2
STATISTICS_FILE = 'statistics.csv'
REPORT_ACCURACY_EVERY = 1
EPOCHS = 20
PATIENCE = 5
BEAM_WIDTH = 3
TEACHER_FORCING_PROPORTION = 0.9
EMBEDDING_DIMENSION = 64
