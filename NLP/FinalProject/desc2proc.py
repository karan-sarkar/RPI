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

BASE_DIR = 'C:\\Users\\Karan Sarkar\\Google Drive\\HFI\\'
EMBEDDING_MATRIX_FILE = BASE_DIR + 'text_word_128.emb'
MODEL_FILE = BASE_DIR + 'desc2proc.pickle'
LOG_FILE = BASE_DIR + 'history.pickle'
TRAIN_DATA_FILE = BASE_DIR + 'desc2proc.csv'
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 128
LSTM_DIM = 128
VALIDATION_SPLIT = 0.1

df = pd.read_csv(TRAIN_DATA_FILE)
trb_nan_idx = df[pd.isnull(df['diag_desc'])].index.tolist()
df.loc[trb_nan_idx, 'diag_desc'] = ' '
df = shuffle(df)

def name2idx(name):
    return df.columns.get_loc(name) + 1

def create_dicts(column, length):
    tokens = set()
    for row in df.itertuples():
        codes = row[column]
        if(isinstance(codes, str)):
            for code in row[column].split():
                tokens.add(code[:length])
    code2idx = {}
    idx2code = {}
    num_codes = len(tokens)
    for i, code in enumerate(tokens):
        code2idx[code[:length]] = i
        idx2code[i] = code[:length]
    return (code2idx, idx2code, num_codes)
    
def one_hot_encode(code2idx, num_codes, column, length):
    print(num_codes)
    data = np.zeros((len(df.index), num_codes))
    idx = 0
    for row in df.itertuples():
        codes = row[column]
        if(isinstance(codes, str)):
            for code in row[column].split():
                data[idx, code2idx[code[:length]]] = 1
        idx += 1
    return data

(code2idx, idx2code, num_codes) = create_dicts(name2idx('proc_code'), 10)
output = one_hot_encode(code2idx, num_codes, name2idx('proc_code'), 10)
print(output.shape)

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(df['diag_desc'])
sequences = tokenizer.texts_to_sequences(df['diag_desc'])
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH, padding = 'pre')
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

pickle_in = open(EMBEDDING_MATRIX_FILE,"rb")
idx2diag, diag2idx, diagnosis_embeddings = pickle.load(pickle_in)

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if word in diag2idx.keys():
        embedding_matrix[i] = diagnosis_embeddings[diag2idx[word], :]
    elif (word + '0') in diag2idx.keys():
        embedding_matrix[i] = diagnosis_embeddings[diag2idx[word + '0'], :]


earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='min')


embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=True)
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(output.shape[1], activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
history = model.fit(data, output, validation_split=0.1, epochs=1, verbose = 1, callbacks=[earlyStopping])

with open(LOG_FILE, 'wb') as f:
    pickle.dump(history.history, f)

filehandler = open(MODEL_FILE,"wb")
pickle.dump(model,filehandler)

#loss: 0.0032 - acc: 0.9995 - auc: 0.6966 - val_loss: 0.0021 - val_acc: 0.9997 - val_auc: 0.7073

