import networkx as nx
import tqdm
import math
import random
import numpy as np

import matplotlib.pyplot as plt
import pylab


edges = []
with open('out.link-dynamic-simplewiki') as file:
    for line in file:
        tokens = line.split()
        if tokens[2] == '+1':
            edges.append(tokens[0] + ' ' + tokens[1])
        
wiki = nx.parse_edgelist(edges, create_using=nx.DiGraph())
gnutella = nx.read_edgelist('p2p-Gnutella31.txt', create_using=nx.DiGraph())
power = nx.read_edgelist('out.opsahl-powergrid')
enron = nx.read_edgelist('Email-Enron.txt')

def centrality_measures(graph, iters):
    results = np.zeros((len(graph.nodes), iters + 1))
    results[:, 0] = 1
    node2idx = {}
    for node in graph.nodes:
        node2idx[node] = len(node2idx)
    for i in range(iters):
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                results[node2idx[node], i + 1] += results[node2idx[neighbor], i]
    return results

cm = np.log(centrality_measures(enron, 10))
a = cm[:, 1]
b = cm[:, 9]
plt.scatter(a, b)

z = np.polyfit(a.flatten(), b.flatten(), 1)
p = np.poly1d(z)
plt.plot(a,p(a),"r--")
plt.title("y=%.6fx+%.6f"%(z[0],z[1])) 
plt.show()

    