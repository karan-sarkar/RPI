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
DIM = 20
EPOCHS = 10
BATCH_SIZE = 256
ITERATIONS = 3
TEST_RATIO = 0.9

def parse_graphs(file):
    edges = []
    with open(TRAIN_FILE) as file:
        for line in file:
            tokens = line.strip().split()
            edges.append(tokens[0] + ' ' + tokens[1])
            
    random.shuffle(edges)

    threshold = int(len(edges) * TEST_RATIO)
    input_graph = nx.parse_edgelist(edges[:threshold])
    output_graph = nx.parse_edgelist(edges[threshold:])
    return input_graph, output_graph
    
input_train_graph, output_train_graph = parse_graphs(TRAIN_FILE)
input_test_graph, output_test_graph = parse_graphs(TEST_FILE)

def collect_nodes(input_graph, output_graph):
    nodes = set(input_graph.nodes())
    nodes.update(output_graph.nodes())
    nodes = list(nodes)
    node2idx = {nodes[i]:i for i in range(len(nodes))}
    return nodes, node2idx

train_idx2node, train_node2idx = collect_nodes(input_train_graph, output_train_graph)
test_idx2node, test_node2idx = collect_nodes(input_test_graph, output_test_graph)
TRAIN_LEN = len(train_idx2node)
TEST_LEN = len(test_idx2node)

class GRNN(nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()
        self.Wz = nn.Linear(DIM, DIM)
        self.Uz = nn.Linear(DIM, DIM)
        self.Wr = nn.Linear(DIM, DIM)
        self.Ur = nn.Linear(DIM, DIM)
        self.Wh = nn.Linear(DIM, DIM)
        self.Uh = nn.Linear(DIM, DIM)
    
    def forward(self, input_graph, node2depth, h):
        for i in range(ITERATIONS):
            x = {index:torch.zeros(DIM) for index in h.keys() if node2depth[index] + i <= ITERATIONS}
            for u in x.keys():
                if u in input_graph.nodes():
                    for v in input_graph.neighbors(u):
                        if v in x.keys():
                            x[u] += h[v]
            z = {index:(torch.sigmoid(self.Wz(x[index]) + self.Uz(h[index]))) for index in x.keys()}
            r = {index:(torch.sigmoid(self.Wr(x[index]) + self.Ur(h[index]))) for index in x.keys()}
            h = {index:(z[index] * h[index] + (1 - z[index]) * torch.tanh(self.Wh(x[index]) + self.Uh(r[index] * h[index]))) for index in x.keys()}
        return h

model = GRNN()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_edges = list(output_train_graph.edges())
test_edges = list(output_train_graph.edges())

def rand_reighbor(graph, u):
    rand = random.random()
    total = 0
    for v in graph.neighbors(u):
        total += 1 / graph.degree(u)
        if total >= rand:
            return v

for _ in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    pbar = tqdm.tqdm(range(0, len(train_edges), BATCH_SIZE))
    for i in pbar:
        num_batches += 1
        batch = train_edges[i:i+BATCH_SIZE]
        size = len(batch)
        
        node2depth = {}
        
        fake_edges = [rand_reighbor(input_train_graph, edge[1]) for edge in batch if edge[1] in input_train_graph.nodes()]
        for edge in batch:
            node2depth[edge[0]] = 0
            node2depth[edge[1]] = 0
        for fake_edge in fake_edges:
            node2depth[fake_edge] = 0
        
        for iter in range(ITERATIONS):
            temp = set()
            for u in node2depth.keys():
                if u in input_train_graph.nodes():
                    for v in input_train_graph.neighbors(u):
                        temp.add(v)
            for u in temp:
                if u not in node2depth.keys():
                    node2depth[u] = iter + 1
                    
        model.zero_grad()
        input = {index:torch.zeros(DIM)  for index in node2depth.keys()}
        for edge in batch:
            input[edge[0]] = torch.ones(DIM)
        output = model.forward(input_train_graph, node2depth, input)
        
        v = torch.stack([output[edge[1]] for edge in batch])
        real_result = torch.sigmoid(torch.sum(v, dim=1)) 
        real_loss = loss_function(real_result, torch.ones_like(real_result))
        
        v = torch.stack([output[fake_edge] for fake_edge in fake_edges])
        fake_result = torch.sigmoid(torch.sum(v, dim=1)) 
        fake_loss = loss_function(fake_result, torch.zeros_like(fake_result))
        
        loss = (real_loss + fake_loss) / 2
        pbar.set_description("%f" % loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(total_loss / num_batches)