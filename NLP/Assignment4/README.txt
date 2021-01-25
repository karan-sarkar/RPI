Hyperparameters
decoder_hidden_size: 256
embed_dropout: 0.6
decoder_dropout: 0.6
lr: 0.001
grad_clipping: 5.0
gpu: False
device: cpu
threads: 0
val_metric: Bleu_4

RNN Loss No Attention: 4.806785583496094
RNN Loss Attention: 5.012942790985107
LSTM Loss No Attention: 5.078618049621582
LSTM Loss Attention: 5.097651958465576

6.2.1
Multiple reference captions per image allows the neural network to pickup different perspectives on the same picture, giving a better chance of hitting on one of them. 
On the other hand, it create a one to many relationship when neural networks are coding a one to one relationship. Thus, the neural network could get confused by having different outputs and
not know which one to report.

6.2.2
Correct names would be hard to report because proper nouns are so low in frequency due to Zipf's law. There would not be enough training data for a given name. Moreover, it would have to work on the same
person at different ages.