import tensorflow as tf
tf.enable_eager_execution()

import random
import re
import numpy as np
import time
import tqdm

from sklearn.model_selection import train_test_split

DATA_FILE = 'athena_train.txt'
text = open(DATA_FILE, 'rb').read().decode(encoding='utf-8')

data = []
para = ''
start = True
for line in text.splitlines():
    if line == '':
        if not start:
            data.append(para)
            para = ''
    else:
        para += line + '\n'
    start = False

claims = [para.splitlines()[0] for para in data]
claims = [claim[claim.index(':=') + 2:] for claim in claims]
claims = ['{' + claim + '}' for claim in claims]

proofs = [' '.join(para.splitlines()[1:]) for para in data]
proofs = [proof.replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ') for proof in proofs]
keywords = set(['as', 'assume', 'by-induction', 'datatype-cases', 'generalize-over', 'let', 'pick-any', 'pick-witness', 'pick-witnesses', 'with-witness'])
proofs = [proof.split() for proof in proofs]
proofs = [[word for word in proof if (word[0] == '!' or word in keywords)] for proof in proofs]
proofs = ['<start> ' + ' '.join(proof) + ' <end>' for proof in proofs]

def tokenize(lang, on_char):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' \n\t\r\n', char_level = on_char)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

input_tensor, inp_lang = tokenize(claims, True)
target_tensor, targ_lang = tokenize(proofs, False)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1, shuffle = True)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 5
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
val_steps_per_epoch = len(input_tensor_val)//BATCH_SIZE
embedding_dim = 256
units = 200
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

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

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

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

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
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
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

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
  i = 0
  pbar = tqdm.tqdm(enumerate(dataset.take(steps_per_epoch)),  total = steps_per_epoch)
  for (batch, (inp, targ)) in pbar:
    i += 1
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
    pbar.set_description('LOSS : %f' % (total_loss / i))
    
  encoder.save_weights('struct_encoder_' + str(epoch + 1))
  decoder.save_weights('struct_decoder_' + str(epoch + 1))

  enc_hidden = encoder.initialize_hidden_state()
  val_loss = 0
  i = 0
  matches = 0
  total = 0
  pbar = tqdm.tqdm(enumerate(val_dataset.take(val_steps_per_epoch)), total = val_steps_per_epoch)
  for (batch, (inp, targ)) in pbar:
    loss = 0
    i += 1
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    total += np.sum(targ.numpy() != 0)
    temp = 0
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      matches += np.sum(targ[:, t].numpy() == tf.argmax(predictions, 1).numpy())
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    val_loss += batch_loss
    pbar.set_description('LOSS : %f' % (val_loss / i))
    
  print('Epoch {} Loss {:.4f} Test Loss {:.4f} Test Acc {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch, val_loss/val_steps_per_epoch, matches/total))
