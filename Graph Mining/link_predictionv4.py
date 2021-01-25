import numpy as np
import networkx as nx
import tqdm
import random
from collections import Counter
from sklearn.metrics import roc_auc_score
from math import log

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim


TRAIN_FILE = 'com-dblp.ungraph.txt'
TEST_FILE = 'out.facebook-wosn-wall'
DIM = 10
EPOCHS = 10
BATCH_SIZE = 256
ITERATIONS = 2
TEST_RATIO = 0.9

def parse_graphs(file_name):
    edges = []
    with open(file_name) as file:
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
                        
            h_temp = {}
            for index in x.keys():
                x_i = x[index]
                h_i = h[index]
                z = torch.sigmoid(self.Wz(x_i) + self.Uz(h_i))
                r = torch.sigmoid(self.Wr(x_i) + self.Ur(h_i))
                h_temp[index] = z * h_i + (1 - z) * torch.tanh(self.Wh(x_i) + self.Uh(r * h_i))
            h = h_temp
        return h

 
        

model = GRNN()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_edges = list(output_train_graph.edges())
test_edges = list(output_test_graph.edges())

def rand_reighbor(graph, u):
    rand = random.random()
    total = 0
    for v in graph.neighbors(u):
        total += 1 / graph.degree(u)
        if total >= rand:
            return v
            
def rand_node(graph):
    rand = random.random()
    size = len(graph.nodes())
    total = 0
    for v in graph.nodes():
        total += 1 / size
        if total >= rand:
            return v

def run(edges, input_graph, train):
    total_loss = 0
    num_batches = 0
    pbar = tqdm.tqdm(range(0, len(edges), BATCH_SIZE))
    preds = []
    targets = []
    for i in pbar:
        num_batches += 1
        batch = edges[i:i+BATCH_SIZE]
                    
        if train:
            model.zero_grad()
        input = {edge[0]:torch.ones(DIM) for edge in batch}
        for edge in batch:
            input[edge[0]] = torch.ones(DIM)
        output = model.forward(input_graph, input)
        real_output = [torch.sum(output[edge[1]]) for edge in batch]
        fake_output = [torch.sum(output[fake_edge]) for fake_edge in fake_edges]
        
        preds.extend([x.item() for x in real_output])
        preds.extend([x.item() for x in fake_output])
        targets.extend([1 for _ in real_output])
        targets.extend([0 for _ in fake_output])
        
        
        real_result = torch.sigmoid(torch.stack(real_output)) 
        real_loss = loss_function(real_result, torch.ones_like(real_result))
        
        fake_result = torch.sigmoid(torch.stack(fake_output)) 
        fake_loss = loss_function(fake_result, torch.zeros_like(fake_result))
        
        loss = (real_loss + fake_loss) / 2
        pbar.set_description("%f" % loss.item())
        total_loss += loss.item()
        if train:
            loss.backward()
            optimizer.step()
    flag = 'TRAIN' if train else 'TEST'
    print(flag + 'GRU LOSS', total_loss / num_batches)
    print(flag + 'GRU AUC', roc_auc_score(np.array(targets), np.array(preds)))

'''
adar_adamic = []
common_neighbors = []
jaccard = []
labels = []
for edge in tqdm.tqdm(test_edges):
    w = rand_node(input_test_graph)
    while w == edge[0]:
        w = rand_node(input_test_graph)
    if edge[1] in input_test_graph.nodes() and edge[0] in input_test_graph.nodes():
        aa_true = 0
        aa_false = 0
        cn_true = 0
        cn_false = 0
        tn_source = 0
        tn_true = 0
        tn_true = input_test_graph.degree(edge[1])
        tn_false = input_test_graph.degree(w)
        for u in input_test_graph.neighbors(edge[0]):
            tn_source += 1
            if input_test_graph.degree(u) > 1:
                for v in input_test_graph.neighbors(u):
                    if v == w:
                        cn_false += 1
                        aa_false += 1 / log(input_test_graph.degree(u))
                    if v == edge[1]:
                        aa_true += 1 / log(input_test_graph.degree(u))
                        cn_true += 1
        common_neighbors.append(cn_true)
        common_neighbors.append(cn_false)
        jaccard.append(cn_true / (tn_source + tn_true - cn_true))
        jaccard.append(cn_false / (tn_source + tn_false - cn_false))
        adar_adamic.append(aa_true)
        adar_adamic.append(aa_false)
        labels.append(1)
        labels.append(0)

labels = np.array(labels)
print('COMMON-NEIGHBORS AUC', roc_auc_score(labels, np.array(common_neighbors)))
print('JACCARD AUC', roc_auc_score(labels, np.array(jaccard)))
print('ADAR-ADAMIC AUC', roc_auc_score(labels, np.array(adar_adamic)))
'''

    
    

for _ in range(EPOCHS):
    run(train_edges, input_train_graph, True)
    with torch.no_grad():
        run(test_edges, input_test_graph, False)