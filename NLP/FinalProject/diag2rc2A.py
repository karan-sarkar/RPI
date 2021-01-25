import pandas as pd
import numpy as np
from builtins import isinstance
import pickle

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

BASE_DIR = 'C:\\Users\\Karan Sarkar\\Google Drive\\HFI\\'
TRAIN_DATA_FILE = BASE_DIR + 'chpw_member_level.csv'
MODEL_FILE = BASE_DIR + 'diag2rc2.pickle'


df = pd.read_csv(TRAIN_DATA_FILE)
#df['codes'] = df['diag_codes'].map(str) + df['therapeutic_class_codes']


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

(code2idx, idx2code, num_codes) = create_dicts(name2idx('diag_codes'), 3)
x_train = one_hot_encode(code2idx, num_codes, name2idx('diag_codes'), 3)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=0, verbose=1, epsilon=1e-4, mode='min', cooldown=1)

model = Sequential()
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', auc])
model.fit(x_train, np.array(df['rc2']), validation_split=0.1, epochs=20, verbose = 1, callbacks=[earlyStopping])

filehandler = open(MODEL_FILE,"wb")
pickle.dump(((code2idx, idx2code, num_codes), model) ,filehandler)

