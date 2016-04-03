import copy
import os
import time
from collections import namedtuple

import numpy as np

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, TRAIN_BATCH_SIZE, TEST_DATASET_PATH, NN_MODEL_PATH, FULL_LEARN_ITER_NUM
from lib.nn_model.predict import predict_sentence
from lib.w2v_model.vectorizer import get_token_vector
from utilities.utils import get_logger

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_logger = get_logger(__name__)


def log_predictions(sentences, nn_model, w2v_model, index_to_token, stats_info=None):
    for sent in sentences:
        prediction = predict_sentence(sent, nn_model, w2v_model, index_to_token)
        _logger.info('[%s] -> [%s]' % (sent, prediction))


def get_test_senteces(file_path):
    with open(file_path) as test_data_fh:
        test_sentences = test_data_fh.readlines()
        test_sentences = [s.strip() for s in test_sentences]

    return test_sentences


def _batch(tokenized_dialog_lines_en,tokenized_dialog_lines_de, batch_size=2):
    batch = []

    for line_en,line_de in zip(tokenized_dialog_lines_en,tokenized_dialog_lines_de):
        batch.append(line_en)
        batch.append(line_de)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # return an empty array instead of yielding incomplete batch
    yield []


def get_training_batch(w2v_model, tokenized_dialog_en,tokenized_dialog_de, token_to_index_de):
    token_voc_size = len(token_to_index_d   e)

    for sents_batch in _batch(tokenized_dialog_en,tokenized_dialog_de, SAMPLES_BATCH_SIZE):
        print "sents_batch: ", np.shape(sents_batch)
        if not sents_batch:
            continue

        X = np.zeros((len(sents_batch)/2, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float)
        Y = np.zeros((len(sents_batch)/2, ANSWER_MAX_TOKEN_LENGTH, token_voc_size), dtype=np.bool)
        # for s_index, sentence in enumerate(sents_batch):
        for s_index in range(0, len(sents_batch),2):
            print "s_index: ",s_index
            # print "s_s_index",s_s_index
            if s_index == len(sents_batch) - 1:
                break

            for t_index, token in enumerate(sents_batch[s_index][:INPUT_SEQUENCE_LENGTH]):
                X[s_index/2, t_index] = get_token_vector(token, w2v_model)

            for t_index, token in enumerate(sents_batch[s_index + 1][:ANSWER_MAX_TOKEN_LENGTH]):
                Y[s_index/2, t_index, token_to_index_de[token]] = 1

            # print "SHAPES X and Y:",np.shape(X),np.shape(Y)

        print X
        print '------------'
        print Y
        yield X, Y


def save_model(nn_model):
    model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    nn_model.save_weights(model_full_path, overwrite=True)


def train_model(nn_model, w2v_model_en, tokenized_dialog_lines_en, tokenized_dialog_lines_de, index_to_token_en, index_to_token_de):
    token_to_index_de = dict(zip(index_to_token_de.values(), index_to_token_de.keys()))
    test_sentences = get_test_senteces(TEST_DATASET_PATH)

    print "STARTED"

    start_time = time.time()
    sents_batch_iteration = 1

    for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
        _logger.info('Full-data-pass iteration num: ' + str(full_data_pass_num))
        dialog_lines_for_train_en = copy.copy(tokenized_dialog_lines_en)
        dialog_lines_for_train_de = copy.copy(tokenized_dialog_lines_de)

        for X_train, Y_train in get_training_batch(w2v_model_en, dialog_lines_for_train_en, dialog_lines_for_train_de, token_to_index_de):
            nn_model.fit(X_train, Y_train, batch_size=TRAIN_BATCH_SIZE, nb_epoch=1, show_accuracy=True, verbose=1)
            print "FIT DONE"

            # if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
            #     log_predictions(test_sentences, nn_model, w2v_model_en, index_to_token_de)
            #     save_model(nn_model)

            sents_batch_iteration += 1

        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
    save_model(nn_model)
