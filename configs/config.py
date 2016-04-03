import os
import time

# set paths for storing models and results locally
DATA_PATH = 'saved/'
CORPORA_DIR = 'corpora_raw'
PROCESSED_CORPORA_DIR = 'corpora_processed'
W2V_MODELS_DIR = 'w2v_models'

# set paths of training and testing sets
# CORPUS_NAME = 'movie_lines_cleaned_10k'
# CORPUS_PATH = os.path.join('data/train', CORPUS_NAME + '.txt')

CORPUS_NAME ="europarl-v7"

# CORPUS_NAME_EN = 'europarl-v7.de-en-64.en'
# CORPUS_NAME_DE = 'europarl-v7.de-en-64.de'

CORPUS_NAME_EN = 'movie_lines_cleaned_10k-64.txt'
CORPUS_NAME_DE = 'movie_lines_cleaned_10k-64.txt'

CORPUS_PATH_EN = os.path.join('data/train', CORPUS_NAME_EN)
CORPUS_PATH_DE = os.path.join('data/train', CORPUS_NAME_DE)

TEST_DATASET_PATH = os.path.join('data', 'test', 'test_set.txt')

# set word2vec params
TOKEN_REPRESENTATION_SIZE = 64  #w2v model dimension
TOKEN_MIN_FREQUENCY = 5

#set seq2seq params
HIDDEN_LAYER_DIMENSION = 128	
INPUT_SEQUENCE_LENGTH = 16			
ANSWER_MAX_TOKEN_LENGTH = 6		

# set training params
TRAIN_BATCH_SIZE = 64
SAMPLES_BATCH_SIZE = TRAIN_BATCH_SIZE
TEST_PREDICTIONS_FREQUENCY = 5			
FULL_LEARN_ITER_NUM = 500

# local paths and strs that depend on previous params
TOKEN_INDEX_PATH_EN = os.path.join(DATA_PATH, 'words_index', 'w_idx_' + CORPUS_NAME_EN + '_m' + str(TOKEN_MIN_FREQUENCY) + '.txt')
TOKEN_INDEX_PATH_DE = os.path.join(DATA_PATH, 'words_index', 'w_idx_' + CORPUS_NAME_DE + '_m' + str(TOKEN_MIN_FREQUENCY) + '.txt')
PROCESSED_CORPUS_PATH_EN = os.path.join(DATA_PATH, PROCESSED_CORPORA_DIR, CORPUS_NAME_EN + '_m' + str(TOKEN_MIN_FREQUENCY) + '.txt')
PROCESSED_CORPUS_PATH_DE = os.path.join(DATA_PATH, PROCESSED_CORPORA_DIR, CORPUS_NAME_DE + '_m' + str(TOKEN_MIN_FREQUENCY) + '.txt')

DATE_INFO = time.strftime('_%d_%H_%M_')
NN_MODEL_PARAMS_STR = '_' + CORPUS_NAME + '_w' + str(TOKEN_REPRESENTATION_SIZE) + '_l' + str(HIDDEN_LAYER_DIMENSION) + \
                      '_s' + str(INPUT_SEQUENCE_LENGTH) + '_b' + str(TRAIN_BATCH_SIZE) + '_m' + str(TOKEN_MIN_FREQUENCY)

# w2v params that depend on previous params
W2V_PARAMS_EN = {
    "corpus_name": CORPUS_NAME_EN,
    "save_path": DATA_PATH,
    "pre_corpora_dir": CORPORA_DIR,
    "new_models_dir": W2V_MODELS_DIR,
    "vect_size": TOKEN_REPRESENTATION_SIZE,
    "min_w_num": TOKEN_MIN_FREQUENCY,
    "win_size": 5,
    "workers_num": 25
}

W2V_PARAMS_DE = {
    "corpus_name": CORPUS_NAME_DE,
    "save_path": DATA_PATH,
    "pre_corpora_dir": CORPORA_DIR,
    "new_models_dir": W2V_MODELS_DIR,
    "vect_size": TOKEN_REPRESENTATION_SIZE,
    "min_w_num": TOKEN_MIN_FREQUENCY,
    "win_size": 5,
    "workers_num": 25
}

# nn params that depend on previous params
NN_MODEL_NAME = 'seq2seq' + NN_MODEL_PARAMS_STR
NN_MODEL_PATH = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_NAME)
