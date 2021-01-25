import numpy as np
import networkx as nx
import tqdm
import random
from collections import Counter
from sklearn.metrics import roc_auc_score

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim


TRAIN_FILE = 'out.facebook-wosn-wall'
TEST_FILE = 'Email-Enron.txt'
DIM = 100
EPOCHS = 10
BATCH_SIZE = 256
LAYERS = 3

edges = []
with open(TRAIN_FILE) as file:
    for line in file:
        tokens = line.strip().split()
        edges.append(tokens[0] + ' ' + tokens[1])

train_graph = nx.parse_edgelist(edges)
test_graph = nx.read_edgelist(TEST_FILE)

TRAIN_LEN = len(train_graph.nodes)
TEST_LEN = len(test_graph.nodes)

train_nodes = list(train_graph.nodes())
train_node2idx = dict((train_nodes[i], i) for i in range(len(train_nodes)))
test_nodes = list(test_graph.nodes())
test_node2idx = dict((test_nodes[i], i) for i in range(len(test_nodes)))

train_edges = list(train_graph.edges())
train_edges = [[train_node2idx[edge[0]], train_node2idx[edge[1]]] for edge in train_edges]
test_edges = list(test_graph.edges())
test_edges = [[test_node2idx[edge[0]], test_node2idx[edge[1]]] for edge in test_edges]

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.weights = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])
    
    def forward(self, graph, node2idx, data, nodes):
        i = 0
        for i in range(LAYERS):
            W = self.weights[i]
            i += 1
            old_data = data
            data = {node:torch.zeros(DIM) for node in old_data.keys()}
            for u, v in graph.edges():
                a = node2idx[u]
                b = node2idx[v]
                if a in nodes.keys() and b in nodes.keys():
                    if nodes[a] + i <= LAYERS:
                        data[a] += W(old_data[b])
                    if nodes[b] + i <= LAYERS:
                        data[b] += W(old_data[a])
            if i != LAYERS:
                data = {node:torch.relu(data[node]) for node in data.keys()}
        return data

gcn = GCN()
loss_function = nn.BCELoss()
optimizer = optim.Adam(gcn.parameters(), lr=0.001)

for _ in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    pbar = tqdm.tqdm(range(0, len(train_edges), BATCH_SIZE))
    for i in pbar:
        num_batches += 1
        batch = train_edges[i:i+BATCH_SIZE]
        size = len(batch)
        
        nodes = {}
        fake_edges = torch.LongTensor(size, 2).random_(0, TRAIN_LEN)
        for edge in batch:
            nodes[edge[0]] = 0
            nodes[edge[1]] = 0
        for i in range(size):
            nodes[fake_edges[i, 0].item()] = 0
            nodes[fake_edges[i, 1].item()] = 0
        
        for layer in range(LAYERS):
            temp = set()
            for i in nodes.keys():
                for v in train_graph.neighbors(train_nodes[i]):
                    temp.add(train_node2idx[v])
            for i in temp:
                if i not in nodes.keys():
                    nodes[i] = layer + 1
        
        
        
        gcn.zero_grad()
        def encode(node):
            return torch.randn(DIM) if nodes[node] == 0 else torch.zeros(DIM)
        input = {node:encode(node)  for node in nodes}
        output = gcn.forward(train_graph, train_node2idx, input, nodes)
        
        u = torch.stack([output[edge[0]] for edge in batch])
        v = torch.stack([output[edge[1]] for edge in batch])
        real_result = torch.sigmoid(torch.bmm(u.view(-1, 1, DIM), v.view(-1, DIM, 1)))
        real_loss = loss_function(real_result, torch.ones_like(real_result))
        
        u = torch.stack([output[fake_edges[i, 0].item()] for i in range(size)])
        v = torch.stack([output[fake_edges[i, 1].item()] for i in range(size)])
        
        fake_result = torch.sigmoid(torch.bmm(u.view(-1, 1, DIM), v.view(-1, DIM, 1)))
        fake_loss = loss_function(fake_result, torch.zeros_like(fake_result))
        
        loss = (real_loss + fake_loss) / 2
        pbar.set_description("%f" % loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(total_loss / num_batches)
    total_loss = 0
    num_batches = 0
    pbar = tqdm.tqdm(range(0, len(test_edges), BATCH_SIZE))
    for i in pbar:
        num_batches += 1
        batch = test_edges[i:i+BATCH_SIZE]
        size = len(batch)
        
        nodes = {}
        fake_edges = torch.LongTensor(size, 2).random_(0, TEST_LEN)
        for edge in batch:
            nodes[edge[0]] = 0
            nodes[edge[1]] = 0
        for i in range(size):
            nodes[fake_edges[i, 0].item()] = 0
            nodes[fake_edges[i, 1].item()] = 0
        
        for layer in range(LAYERS):
            temp = set()
            for i in nodes.keys():
                for v in test_graph.neighbors(test_nodes[i]):
                    temp.add(test_node2idx[v])
            for i in temp:
                if i not in nodes.keys():
                    nodes[i] = layer + 1
        
        for _ in range(LAYERS):
            temp = set()
            for i in nodes:
                for v in test_graph.neighbors(test_nodes[i]):
                    temp.add(test_node2idx[v])
            for i in temp:
                nodes.add(i)
        
        def encode(node):
            return torch.randn(DIM) if nodes[node] == 0 else torch.zeros(DIM)
        input = {node:encode(node)  for node in nodes} 
        output = gcn.forward(test_graph, test_node2idx, input, nodes)
        
        u = torch.stack([output[edge[0]] for edge in batch])
        v = torch.stack([output[edge[1]] for edge in batch])
        real_result = torch.sigmoid(torch.bmm(u.view(-1, 1, DIM), v.view(-1, DIM, 1)))
        real_loss = loss_function(real_result, torch.ones_like(real_result))
        
        u = torch.stack([output[fake_edges[i, 0].item()] for i in range(size)])
        v = torch.stack([output[fake_edges[i, 1].item()] for i in range(size)])
        fake_result = torch.sigmoid(torch.bmm(u.view(-1, 1, DIM), v.view(-1, DIM, 1)))
        fake_loss = loss_function(fake_result, torch.zeros_like(fake_result))
        
        loss = (real_loss + fake_loss) / 2
        pbar.set_description("%f" % loss.item())
        total_loss += loss.item()
    print(total_loss / num_batches)