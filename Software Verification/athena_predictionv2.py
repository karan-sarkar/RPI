import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import random
import unicodedata
import re
import numpy as np
import os
import io
import time
import tqdm

TRAIN_FILE = 'athena_train.txt'
TEST_FILE = 'athena_test.txt'

# Read, then decode for py2 compat.
data = open(TRAIN_FILE, 'rb').read().decode(encoding='utf-8')
data = data.replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ')

train = []
test = []
para = ''
train_flag = True
start = True
for line in data.splitlines():
    if line == '':
        train_flag = (random.random() > 0.05)
        if not start:
            if train_flag:
                train.append(para)
            else:
                test.append(para)
            para = ''
    elif train_flag:
        para += line + '\n'
    else:
        para += line + '\n'
    start = False

train_x = ['<start> '  + item.splitlines()[0] + ' <end>' for item in train]
train_y = ['<start> ' + ' '.join((' '.join(item.splitlines()[1:])).split()) + ' <end>' for item in train]
test_x = ['<start> '  + item.splitlines()[0] + ' <end>' for item in test]
test_y = ['<start> ' + ' '.join((' '.join(item.splitlines()[1:])).split()) + ' <end>' for item in test]

train_x = [' '.join(i for i in line.split() if not (i.isalpha() and len(i)==1)) for line in train_x]
train_y = [' '.join(i for i in line.split() if not (i.isalpha() and len(i)==1)) for line in train_y]
print(train_y[0])

data = ' '.join(i for i in data.split() if not (i.isalpha() and len(i)==1))

lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' \n\t\r\n', lower = False)
lang_tokenizer.fit_on_texts(['<start> ' + data + ' <end>'])


train_x = lang_tokenizer.texts_to_sequences(train_x)

train_y = lang_tokenizer.texts_to_sequences(train_y)
test_x = lang_tokenizer.texts_to_sequences(test_x)
test_y = lang_tokenizer.texts_to_sequences(test_y)

train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, padding='post')
train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, padding='post')[:, :50]
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, padding='post')
test_y = tf.keras.preprocessing.sequence.pad_sequences(test_y, padding='post')[:, :50]



BUFFER_SIZE = len(train_x)
BATCH_SIZE = 5
steps_per_epoch = int(len(train_x)//BATCH_SIZE)
test_steps_per_epoch = int(len(test_x)//BATCH_SIZE)
embedding_dim = 256
units = 1024
vocab_size = len(lang_tokenizer.word_index)+1

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(train_dataset))
print(example_input_batch.shape, example_target_batch.shape)

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits = True)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

EPOCHS = 30

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  pbar = tqdm.tqdm(enumerate(train_dataset.take(steps_per_epoch)), total = steps_per_epoch)
  for (batch, (inp, targ)) in pbar:
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
    pbar.set_description('LOSS : %f' % batch_loss)
                                                   
  test_loss = 0
  for (batch, (inp, targ)) in tqdm.tqdm(enumerate(test_dataset.take(test_steps_per_epoch)), total = test_steps_per_epoch):
    loss = 0
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    test_loss += batch_loss

    
  # saving (checkpoint) the model every 2 epochs
  encoder.save_weights('struct_encoder_' + str(epoch))
  decoder.save_weights('struct_decoder_' + str(epoch))

  print('Epoch {} Loss {:.4f} Test Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch, test_loss/test_steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


