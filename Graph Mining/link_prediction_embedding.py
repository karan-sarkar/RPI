import numpy as np
import tqdm
import random
from collections import Counter
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

FILENAME = 'out.facebook-wosn-wall'
OUTPUT_FILE = 'lpe'
EMBED_DIM = 100
HIDDEN_DIM = 100
EPOCHS = 10
BATCH_SIZE = 128
SAMPLES = 100000
SUPPORT = 0
TEST_SIZE = 10000

data = []
word2idx = {}
idx2word = []
counts= []

with open(FILENAME) as file:
    for line in file:
        tokens = line.strip().split()[:2]
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = len(idx2word)
                idx2word.append(token)
                counts.append(1)
            counts[word2idx[token]] += 1
        data.append((word2idx[tokens[0]],  word2idx[tokens[0]]))

counts = np.array(counts)
counts = counts / np.sum(counts)
counts = np.cumsum(counts)

def rand_word(num):
    seed = np.random.rand(num)
    return np.searchsorted(counts, seed, side = 'left')

random.shuffle(data)
NUM_DATA = len(data)
THRESHOLD = NUM_DATA - TEST_SIZE
VOCAB_SIZE = len(word2idx)

class ArrayDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, index):
        u =  torch.from_numpy(np.array(self.data[index][0])).long()
        v = torch.from_numpy(np.array(self.data[index][1])).long()
        l = torch.from_numpy(np.array(self.labels[index])).float()
        return u, v, l
    
    def __len__(self):
        return len(self.data)

def make_train_dataset(data):
    labels = [] 
    new_data = []
    for u, v in data:
        new_data.append((u, v))
        new_data.append((u, int(rand_word(1))))
        new_data.append((v, u))
        new_data.append((v, int(rand_word(1))))
        labels.extend([1, 0, 1, 0])
    return DataLoader(ArrayDataset(new_data, labels), batch_size=BATCH_SIZE, shuffle=True), (new_data, labels) 

def make_test_dataset(data):
    labels = [] 
    new_data = []
    for u, v in data:
        new_data.append((u, v))
        temp = int(rand_word(1))
        while (u, temp) in train_set or (u, temp) in test_set:
            temp = int(rand_word(1))
        new_data.append((u, temp))
        new_data.append((v, u))
        temp = int(rand_word(1))
        while (v, temp) in train_set or (v, temp) in test_set:
            temp = int(rand_word(1))
        new_data.append((v, temp))
        labels.extend([1, 0, 1, 0])
    return DataLoader(ArrayDataset(new_data, labels), batch_size=BATCH_SIZE, shuffle=True), (new_data, labels) 

train_set = set(data[:THRESHOLD])
test_set = set(data[THRESHOLD:])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.U = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.V = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    
    def forward(self, center, context):
        center_embed = self.U(center)
        context_embed = self.V(context)
        out = torch.sigmoid(torch.bmm(center_embed.view(-1, 1, EMBED_DIM), context_embed.view(-1, EMBED_DIM, 1)))
        return out.view(-1)


print(VOCAB_SIZE, len(data))

accs = [1000]
loss_function = nn.BCELoss()

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for _ in range(EPOCHS):
    train_data, _ = make_train_dataset(data[:THRESHOLD])
    for u, v, l in tqdm.tqdm(train_data):
        model.zero_grad()
        output = model.forward(u, v)
        loss = loss_function(output, l)
        loss.backward()
        optimizer.step()
    
    test_data, (_, labels) = make_test_dataset(data[THRESHOLD:])
    total = 0
    total_loss = 0
    results = []
    for u, v, l in tqdm.tqdm(test_data):
        output = model.forward(u, v)
        loss = loss_function(output, l)
        total += 1
        total_loss += loss.item()
        results.extend(output.detach().numpy().tolist())
    
    auc = roc_auc_score(np.array(labels), np.array(results))
    acc = total_loss / total
    print(acc, auc)
    if acc > min(accs):
        break
    accs.append(acc)
    
    u = model.U(torch.tensor(np.arange(VOCAB_SIZE), dtype=torch.long)).detach().numpy()
    v = model.V(torch.tensor(np.arange(VOCAB_SIZE), dtype=torch.long)).detach().numpy()
    embeddings = (u + v) / 2
    with open(OUTPUT_FILE, 'w') as file:
        file.write('%d %d\n' % (VOCAB_SIZE, EMBED_DIM))
         
        for i in range(VOCAB_SIZE):
            s = idx2word[i]
            for j in range(EMBED_DIM):
                s += ' ' + str(embeddings[i, j])
            s += '\n'
            file.write(s)