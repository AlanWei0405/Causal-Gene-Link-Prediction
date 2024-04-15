import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt


def build_networks(ppi_data, dd_data, dg_data):
    # Construct the Gene-Gene Network
    ppi_graph = nx.from_pandas_edgelist(ppi_data, 'Gene 1', 'Gene 2', create_using=nx.Graph())

    # Construct the Disease-Disease Network
    dd_graph = nx.from_pandas_edgelist(dd_data, 'Disease ID1', 'Disease ID2', create_using=nx.Graph())

    # Filter the Disease-Gene Data
    # Filter diseases and genes based on what's available in disease-disease and gene-gene networks
    filtered_dg_data = dg_data[(dg_data['Disease ID'].isin(dd_graph.nodes)) &
                               (dg_data['Gene ID'].isin(ppi_graph.nodes))]

    # Construct the Disease-Gene Network
    dg_graph = nx.from_pandas_edgelist(filtered_dg_data, 'Disease ID', 'Gene ID', create_using=nx.Graph())

    # Initialize an integrated network
    integrated_network = nx.Graph()

    # Add the disease-disease and gene-gene interactions
    integrated_network.add_edges_from(dd_graph.edges(data=True))
    integrated_network.add_edges_from(ppi_graph.edges(data=True))

    # Now add the disease-gene interactions
    integrated_network.add_edges_from(filtered_dg_data[['Disease ID', 'Gene ID']].itertuples(index=False, name=None))
    print(ppi_graph)
    print(dd_graph)
    print(dg_graph)
    print(integrated_network)


