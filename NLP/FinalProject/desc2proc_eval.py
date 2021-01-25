from gensim.models import Word2Vec
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from keras.layers.core import Layer
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.utils import class_weight
from sklearn.utils import shuffle

import nltk
import numpy as np
import pandas as pd
import pickle
import math
from gc import callbacks
import keras.metrics

BASE_DIR = 'C:\\Users\\Karan Sarkar\\Google Drive\\HFI\\'
EMBEDDING_MATRIX_FILE = BASE_DIR + 'embedding_matrix.pickle'
MODEL_FILE = BASE_DIR + 'desc2proc.pickle'
TEST_DATA_FILE = BASE_DIR + 'icd.csv'
MAX_SEQUENCE_LENGTH = 200
CONTEXT_WINDOW = 10
MAX_NB_WORDS = 200000
VALIDATION_SPLIT = 0.1

df = pd.read_csv(TEST_DATA_FILE)

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(df['description'])
sequences = tokenizer.texts_to_sequences(df['description'])
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH, padding = 'pre')
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

keras.metrics.auc = auc

pickle_in = open(MODEL_FILE,"rb")
model = pickle.load(pickle_in)
embedding_matrix = model.predict(data, verbose = 1)

code2idx = {}
idx2code = {}
i = 0
for row in df.itertuples():
    code = row['codes']
    code2idx[code] = i
    idx2code[i] = code
    i += 1

filehandler = open(EMBEDDING_MATRIX_FILE,"wb")
pickle.dump((embedding_matrix, code2idx, idx2code),filehandler)