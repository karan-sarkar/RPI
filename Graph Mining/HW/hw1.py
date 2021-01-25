'''
1 a) Both the Wiki data and the Gnutella data are similar to the Web in that they seem to be 
distributed according to a power law. In both cases, there is one dominant strongly connected
component which most nodes are part of or are connected to.

1 b) Wiki has a large IN size and small OUT size. In contrast Gnutella, has a large OUT size 
and small IN size. This is because of the context of the networks. In Wikipedia, most 
articles have a lot of out-edges but only important articles have in-edges. In Gnutella, many
nodes are never never hosts. Thus IN is small.

2 e) We can see that the power law coefficient and hubedness are negatively correlated that is
they move in opposite directions. The power network is very centralized as has a high power 
coefficient but a low level of hubedness. On ther other hand, Wikipedia has low power law
coefficient but a high level of hubedness. The inverse relationship makes sense because
the more hubs there are the less centralized the graph is and the lower the power law.
Wikipedia has a low centralization level since so many articles are interconneted. On the other
hand, the power grid is highly ordered and is very centralized. The power grid also has a much
greater average shortest path than the other networks. This is due to the geographic nature of
the network. Connections can only exist if they make geographic sense.
'''

import networkx as nx
import tqdm
import math
import random


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

def bow_tie(graph, name):
    
    print(name + ': Weakly Connected Components', len(list(nx.weakly_connected_components(graph))))
    print(name + ': Strongly Connected Components', len(list(nx.strongly_connected_components(graph))))
    
    wiki_degree = graph.out_degree(graph.nodes)
    print(name + ': Trivial Strongly Connected Components', sum([degree[1] == 0 for degree in wiki_degree]))
    
    wiki_giant = max(nx.strongly_connected_components(graph), key=len)
    print(name + ': SCC Size', len(wiki_giant))
    wiki_giant_node = next(iter(wiki_giant))
    
    wiki_in = []
    wiki_out = []
    wiki_disconnected = []
    
    for component in nx.strongly_connected_components(graph):
        if component == wiki_giant:
            continue
        node = next(iter(component))
        neither = True
        if nx.has_path(graph, node, wiki_giant_node):
            wiki_in.append(component)
            neither = False
        if nx.has_path(graph, wiki_giant_node, node):
            wiki_out.append(component)
            neither = False
        if neither:
            wiki_disconnected.append(component)
    
    print(name + ': IN Size', sum([len(c) for c in wiki_in]))
    print(name + ': OUT Size', sum([len(c) for c in wiki_out]))
    
    tendrils = []
    tubes = []
    for component in tqdm.tqdm(wiki_disconnected):
        node = next(iter(component))
        in_tendril = False
        for in_component in wiki_in:
            in_node = next(iter(in_component))
            if nx.has_path(graph, in_node, node):
                tendrils.append(len(component))
                in_tendril = True
                break
        for out_component in wiki_out:
            out_node = next(iter(out_component))
            if nx.has_path(graph, node, out_node):
                if in_tendril:
                    tubes.append(len(component))
                else:
                    tendrils.append(len(component))
                break
            
    print(name + ': Tendril Sizes', tendrils)
    print(name + ': Tube Sizes', tubes)
    print(name + ': Tendril Count', sum(tendrils))
    print(name + ': Tube Count', sum(tubes))

def biconnect(graph, name):
    vertices = []
    for node in tqdm.tqdm(graph.nodes):
        new_graph = graph.copy()
        new_graph.remove_node(node)
        vertices.append((node, len(max(nx.connected_components(new_graph), key=len))))
    print(name + ': Central Vertex', min(vertices, key=lambda x:x[1]))

def measure(graph, name):
    deg = 0
    hubs = 0
    for node in graph.nodes:
        deg += math.log(graph.degree(node))
        if graph.degree(node) > math.log(len(graph.nodes)):
            hubs += 1
    alpha = 1 + len(graph.nodes) / deg
    print(name + ": Power Law Coefficient", alpha)  
    print(name + ": Hubedness", hubs / len(graph.nodes))   
    
    cc = None
    try:
        cc = max(nx.connected_components(graph), key=len)
    except:
        cc = max(nx.strongly_connected_components(graph), key=len)
    
    sample = random.sample(cc, 101)
    start = sample.pop()
    total = 0
    for end in sample:
        total += nx.shortest_path_length(graph, start, end)
    print(name + ": Average Shortest Path", total / 100)
            
    
    

bow_tie(wiki, 'Wiki')
bow_tie(gnutella, 'Gnutella')
biconnect(power, 'Power')
biconnect(enron, 'Enron')
measure(wiki, 'Wiki')
measure(gnutella, 'Gnutella')
measure(power, 'Power')
measure(enron, 'Enron')

    

