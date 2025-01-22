import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# np.random.seed(7)

def generate_ppi_network_data(num_nodes, edge_density=0.2):
    nodes = np.arange(num_nodes)
    edges = []

    mode = "sw"

    if mode == "sw":
        # Parameters for Watts-Strogatz
        n = num_nodes  # Number of nodes in each set
        k = 6 # Each node is connected to K nearest neighbors in its own set
        p = 0.2  # Rewiring probability

        # Generate two separate Watts-Strogatz graphs for the two node sets
        g = nx.newman_watts_strogatz_graph(n, k, p)
        print(g.number_of_edges())

        # # Get the degree of each node
        # degrees = [g.degree(n) for n in g.nodes()]
        #
        # # Plot the histogram of node degrees
        # plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor='black', align='left')
        #
        # # Add labels and title
        # plt.xlabel('Degree')
        # plt.ylabel('Frequency')
        # plt.title('Node Degree Distribution')
        #
        # # Show the plot
        # plt.show()

        edges=g.edges()
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_density:
                    edges.append([nodes[i], nodes[j]])

    return pd.DataFrame(edges, columns=['g1', 'g2'])

def generate_dd_network_data(num_nodes, edge_density=0.2):
    nodes = np.arange(num_nodes)
    edges = []

    mode = "sw"

    if mode == "sw":
        # Parameters for Watts-Strogatz
        n = num_nodes  # Number of nodes in each set
        k = 5 # Each node is connected to K nearest neighbors in its own set
        p = 0.1  # Rewiring probability

        # Generate two separate Watts-Strogatz graphs for the two node sets
        g = nx.newman_watts_strogatz_graph(n, k, p)
        print(g.number_of_edges())
        edges=g.edges()
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_density:
                    edges.append([nodes[i], nodes[j]])

    return pd.DataFrame(edges, columns=['d1', 'd2'])

def generate_dg_data(num_genes, num_diseases, edge_density=0.0005):
    genes = np.arange(num_genes)
    diseases = np.arange(num_diseases)
    edges = []

    for g in genes:
        for d in diseases:
            if np.random.rand() < edge_density:
                edges.append([g, d + num_genes])  # Disease nodes are offset by num_genes

    print(len(edges))

    return pd.DataFrame(edges, columns=['g', 'd'])
