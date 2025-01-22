import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
import pickle

from matplotlib import pyplot as plt
from torch_geometric.data import Data, HeteroData



def build_networks(ppi_data, dd_data, dg_data):

    ppi_nt_path = './ppi_nt.pkl'
    dd_nt_path = './dd_nt.pkl'
    dg_nt_path = './dg_nt.pkl'
    dgt_nt_path = './dgt_nt.pkl'
    all_nt_path = './all_nt.pkl'
    hetero_nt_path = './hetero_nt.pkl'

    if (((os.path.exists(ppi_nt_path) and os.path.exists(dd_nt_path)) and
            (os.path.exists(dg_nt_path) and os.path.exists(dgt_nt_path)))) and os.path.exists(all_nt_path):

        with open(ppi_nt_path, 'rb') as f:
            ppi_nt = pickle.load(f)

        with open(dd_nt_path, 'rb') as f:
            dd_nt = pickle.load(f)

        with open(dg_nt_path, 'rb') as f:
            dg_nt = pickle.load(f)

        with open(dgt_nt_path, 'rb') as f:
            dgt_nt = pickle.load(f)

        with open(all_nt_path, 'rb') as f:
            all_nt = pickle.load(f)

        with open(hetero_nt_path, 'rb') as f:
            hetero_data = pickle.load(f)

    else:

        # Construct the Gene-Gene Network
        ppi_graph = nx.from_pandas_edgelist(ppi_data, 'g1', 'g2', create_using=nx.Graph())
        # Create the feature matrix
        ppi_features_df = feature_matrix(ppi_graph)
        print(ppi_graph.number_of_nodes())
        print(ppi_graph.number_of_edges())

        # Construct the Disease-Disease Network
        dd_graph = nx.from_pandas_edgelist(dd_data, 'd1', 'd2', create_using=nx.Graph())
        dd_features_df = feature_matrix(dd_graph)
        print(dd_graph.number_of_nodes())
        print(dd_graph.number_of_edges())

        # plot_degree(dd_graph)

        # Construct the Disease-Gene Network
        dg_graph = nx.Graph()
        mapping = {node: node + ppi_graph.number_of_nodes() for node in dd_graph.nodes()}
        dd_graph_remap = nx.relabel_nodes(dd_graph, mapping)
        dg_graph.add_nodes_from(ppi_graph.nodes)
        dg_graph.add_nodes_from(dd_graph_remap.nodes)
        dg_graph.add_edges_from(list(dg_data[['g', 'd']].itertuples(index=False, name=None)))
        print(dg_graph.number_of_nodes())
        print(dg_graph.number_of_edges())

        # plot_degree(dg_graph)

        # dg_graph = nx.from_pandas_edgelist(dg_data, 'Gene', 'Disease', create_using=nx.Graph())
        dg_features_df = feature_matrix(dg_graph)

        # # Initialize an integrated network
        all_graph = nx.Graph()
        #
        # Add the disease-disease and gene-gene interactions
        all_graph.add_edges_from(dd_graph_remap.edges(data=True))
        all_graph.add_edges_from(ppi_graph.edges(data=True))
        all_graph.add_edges_from(dg_graph.edges(data=True))
        print(all_graph.number_of_nodes())
        print(all_graph.number_of_edges())

        all_features_df = feature_matrix(all_graph)

        all_edge_index = torch.tensor(list(all_graph.edges), dtype=torch.long).t().contiguous()
        all_x = torch.tensor(all_features_df, dtype=torch.float)
        all_nt = Data(x=all_x, edge_index=all_edge_index)

        ppi_edge_index = torch.tensor(ppi_data[['g1', 'g2']].values, dtype=torch.long).t().contiguous()
        ppi_x = torch.tensor(ppi_features_df, dtype=torch.float)
        ppi_nt = Data(x=ppi_x, edge_index=ppi_edge_index)

        dd_edge_index = torch.tensor(dd_data[['d1', 'd2']].values, dtype=torch.long).t().contiguous()
        dd_x = torch.tensor(dd_features_df, dtype=torch.float)
        dd_nt = Data(x=dd_x, edge_index=dd_edge_index)

        dg_edge_index = torch.tensor(dg_data[['g', 'd']].values, dtype=torch.long).t().contiguous()
        dg_x = torch.tensor(dg_features_df, dtype=torch.float)
        dg_nt = Data(x=dg_x, edge_index=dg_edge_index)

        dgt_nt = dg_nt.clone()

        # Initialize HeteroData
        hetero_data = HeteroData()

        # Add nodes (you can also add features if available)
        hetero_data['gene'].x = ppi_x
        hetero_data['disease'].x = dd_x

        # Define edges (use edge_index to represent connectivity)
        # Gene-Gene edges
        hetero_data['gene', 'interact', 'gene'].edge_index = ppi_edge_index
        # Gene-Disease edges
        hetero_data['gene', 'associate', 'disease'].edge_index = dg_edge_index
        # Disease-Disease edges
        hetero_data['disease', 'relate', 'disease'].edge_index = dd_edge_index

        with open(ppi_nt_path, 'wb') as f:
            pickle.dump(ppi_nt, f)

        with open(dd_nt_path, 'wb') as f:
            pickle.dump(dd_nt, f)

        with open(dg_nt_path, 'wb') as f:
            pickle.dump(dg_nt, f)

        with open(dgt_nt_path, 'wb') as f:
            pickle.dump(dgt_nt, f)

        with open(all_nt_path, 'wb') as f:
            pickle.dump(all_nt, f)

        with open(hetero_nt_path, 'wb') as f:
            pickle.dump(hetero_data, f)

    return ppi_nt, dd_nt, dg_nt, dgt_nt, all_nt, hetero_data


def feature_matrix(graph):

    num_nodes = graph.number_of_nodes()
    embedding_dim = 128

    # Generate random embeddings from a normal distribution
    np.random.seed(45)  # Setting seed for reproducibility
    # random_embeddings = np.random.randn(num_nodes, embedding_dim)
    random_embeddings = np.ones((num_nodes, embedding_dim))
    features_df = pd.DataFrame(random_embeddings, index=list(graph.nodes()))

    return features_df.to_numpy()

# def plot_degree(graph):
#     # Get the degree of each node
#     degrees = [graph.degree(n) for n in graph.nodes()]
#
#     # Plot the histogram of node degrees
#     plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor='black', align='left')
#
#     # Add labels and title
#     plt.xlabel('Degree')
#     plt.ylabel('Frequency')
#     plt.title('Node Degree Distribution')
#
#     # Show the plot
#     plt.show()