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
data = ' '.join(i for i in data.split() if not (i.isalpha() and len(i)==1))

lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' \n\t\r\n', lower = False)
lang_tokenizer.fit_on_texts(['<start> ' + data + ' <end>'])

#sent = 'conclude diff-characterization-1 := (forall A B x . x in A \ B ==> x in A & ~ x in B)'
#sent = 'conclude range-theorem-2 := (forall R1 R2 . range (R1 /\ R2) subset range R1 /\ range R2)'
#sent = 'conclude of-join := (forall L M x . (product L ++ M) = (product L) * (product M))'
#sent = 'conclude restriction-theorem-4 := (forall R A B . R ^ (A \ B) = R ^ A \ R ^ B)'
sent = 'conclude sum-lemma2 := (forall x y z . x divides y & x divides (y + z) ==> x divides z)'

sent = '<start> ' + sent + ' <end>'
sent = sent.replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ')
sent = ' '.join(i for i in sent.split() if not (i.isalpha() and len(i)==1))

test_x = [lang_tokenizer.word_index[i] for i in sent.split()]
test_x = tf.keras.preprocessing.sequence.pad_sequences([test_x], maxlen=42, padding='post')


BUFFER_SIZE = len(test_x)
BATCH_SIZE = 1
steps_per_epoch = int(len(test_x)//BATCH_SIZE)
embedding_dim = 256
units = 1024
vocab_size = len(lang_tokenizer.word_index)+1




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

encoder.load_weights('struct_encoder_3')
decoder.load_weights('struct_decoder_3')

test_x = tf.convert_to_tensor(test_x)

enc_hidden = [tf.zeros((1, units))]
enc_output, enc_hidden = encoder(test_x, enc_hidden)

print(enc_hidden)

dec_hidden = enc_hidden
result = ''
dec_input = tf.expand_dims([lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

# Teacher forcing - feeding the target as the next input
for t in range(1, 50):
  # passing enc_output to the decoder
  predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
  
  predicted_id = tf.argmax(predictions[0]).numpy()

  result += lang_tokenizer.index_word[predicted_id] + ' '

  if lang_tokenizer.index_word[predicted_id] == '<end>':
    break

  # the predicted ID is fed back into the model
  dec_input = tf.expand_dims([predicted_id], 0)

  # using teacher forcing
  dec_input = tf.expand_dims([predicted_id], 1)

print(result)

