import tensorflow as tf
tf.enable_eager_execution()

import random
import re
import numpy as np
import time
import tqdm
from collections import OrderedDict 
import math

from sklearn.model_selection import train_test_split

DATA_FILE = 'athena_train.txt'
text = open(DATA_FILE, 'rb').read().decode(encoding='utf-8')
#inputs = ['conclude diff-characterization-1 := (forall A B x . x in A \ B ==> x in A & ~ x in B)']
#inputs = ['conclude of-join := (forall L M x . (product L ++ M) = (product L) * (product M))']
inputs = ['conclude range-theorem-2 := (forall R1 R2 . range (R1 /\ R2) subset range R1 /\ range R2)']
#inputs = ['conclude sum-lemma2 := (forall x y z . x divides y & x divides (y + z) ==> x divides z)']
#inputs = ['conclude dom-corrolary-1 := (forall key val k rest  . k in dom rest - key ==> k in dom [key val] ++ rest)']


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

inputs = [input[input.index(':=') + 2:] for input in inputs]
inputs = ['{' + input + '}' for input in inputs]

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

freqs = targ_lang.sequences_to_matrix(targ_lang.texts_to_sequences(proofs), mode='count') 
freqs = np.sum(freqs, 0) / np.sum(freqs)
print(np.sum(freqs ** 2))

freqs = np.log(freqs)

BATCH_SIZE = 1
embedding_dim = 256
units = 100
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

tensor = inp_lang.texts_to_sequences(inputs)
tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

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

encoder.load_weights('struct_encoder_7')
decoder.load_weights('struct_decoder_7')


enc_hidden = [tf.zeros((1, units))]
enc_output, enc_hidden = encoder(tensor, enc_hidden)
idx = {}
vals = []

outputs = OrderedDict()
for _ in tqdm.tqdm(range(1000)):

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    result = '<start> '
    prob = 0
    prior = 0
    for t in range(1, 50):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      predictions = tf.log(tf.nn.softmax(predictions))
      predicted_id = tf.random.categorical(predictions, num_samples = 1)[0, 0].numpy()
      prob += float(predictions[0, predicted_id].numpy())
      prior += float(freqs[predicted_id])
      if predicted_id not in targ_lang.index_word:
        continue
      
      result += targ_lang.index_word[predicted_id] + ' '
      if targ_lang.index_word[predicted_id] == '<end>':
        break

      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)

      # using teacher forcing
      dec_input = tf.expand_dims([predicted_id], 1)
    length = len(result.split()) - 2
    if length > 0 and not math.isinf(prob) and not math.isinf(prior):
        #idx[result] = len(idx)
        #vals.append([prob, prob - 0.2 * prior])
        outputs[prob - 0.2 * prior] = result

#vals = np.array(vals)
#vals = (vals - np.mean(vals, 0)) / np.std(vals, 0)
#vals[:, 1] = 0.1 * vals[:, 1]
#for result in idx.keys():
 #   outputs[np.sum(vals[idx[result], :])] = result

outputs = OrderedDict(sorted(outputs.items()))
for key, value in outputs.items():
    print(key, value)