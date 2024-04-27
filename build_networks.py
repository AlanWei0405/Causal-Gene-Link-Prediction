import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data


def build_networks(ppi_data, dd_data, dg_data):
    # Construct the Gene-Gene Network
    ppi_graph = nx.from_pandas_edgelist(ppi_data, 'Gene 1', 'Gene 2', create_using=nx.Graph())

    # Create the feature matrix
    ppi_features_df = feature_matrix(ppi_graph)

    # Normalize features scaler = MinMaxScaler()
    # features_df = pd.DataFrame(scaler.fit_transform(features_df),
    # index=features_df.index, columns=features_df.columns)

    # Display the feature matrix
    # pd.set_option('display.precision', 5)
    # pd.set_option('display.max_rows', 1000)
    # print(features_df)

    # Construct the Disease-Disease Network
    dd_graph = nx.from_pandas_edgelist(dd_data, 'Disease ID1', 'Disease ID2', create_using=nx.Graph())

    dd_features_df = feature_matrix(dd_graph)

    # Filter the Disease-Gene Data
    # Filter diseases and genes based on what's available in disease-disease and gene-gene networks
    filtered_dg_data = dg_data[(dg_data['Disease ID'].isin(dd_graph.nodes)) &
                               (dg_data['Gene ID'].isin(ppi_graph.nodes))]

    # Construct the Disease-Gene Network
    dg_graph = nx.from_pandas_edgelist(filtered_dg_data, 'Disease ID', 'Gene ID', create_using=nx.Graph())

    dg_features_df = feature_matrix(dg_graph)

    # Initialize an integrated network
    integrated_network = nx.Graph()

    # Add the disease-disease and gene-gene interactions
    integrated_network.add_edges_from(dd_graph.edges(data=True))
    integrated_network.add_edges_from(ppi_graph.edges(data=True))

    # Now add the disease-gene interactions
    integrated_network.add_edges_from(filtered_dg_data[['Disease ID', 'Gene ID']].itertuples(index=False, name=None))

    ppi_edge_index = torch.tensor(ppi_data[['Gene 1', 'Gene 2']].values, dtype=torch.long).t().contiguous()
    ppi_x = torch.tensor(ppi_features_df.to_numpy(), dtype=torch.float)
    ppi_data = Data(x=ppi_x, edge_index=ppi_edge_index)

    # dd_edge_index = torch.tensor(dd_graph.edges.values, dtype=torch.long).t().contiguous()
    # dd_x = torch.tensor(dd_features_df, dtype=torch.float)
    # dd_data = Data(x=dd_x, edge_index=dd_edge_index)
    #
    # dg_edge_index = torch.tensor(dg_graph.edges.values, dtype=torch.long).t().contiguous()
    # dg_x = torch.tensor(dg_features_df, dtype=torch.float)
    # dg_data = Data(x=dg_x, edge_index=dg_edge_index)

    # print("ppi network:")
    # print(ppi_graph)
    # print("dd network:")
    # print(dd_graph)
    # print("dg network:")
    # print(dg_graph)
    # print("integrated network:")
    # print(integrated_network)

    return (ppi_graph, dd_graph, dg_graph, integrated_network,
            ppi_data, dd_data, dg_data)


def feature_matrix(graph):
    # Create the feature matrix
    features_df = pd.DataFrame({
        'Degree': pd.Series(dict(graph.degree())),
        'Clustering': pd.Series(nx.clustering(graph)),
        'PageRank': pd.Series(nx.pagerank(graph))
    })
    features_df.index = list(graph.nodes())
    features_df.fillna(0, inplace=True)

    return features_df
