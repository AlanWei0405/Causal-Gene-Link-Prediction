import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.nn import GCN

from utils.build_networks import build_networks
from run_vgae import run_vgae


def run_recon(dd_data, dg_data, ppi_data, epoch):

    ppi_nt, dd_nt, dg_nt, dgt_nt, all_nt = build_networks(ppi_data, dd_data, dg_data)

    ppi_full_edges = ppi_nt.edge_index
    dd_full_edges = dd_nt.edge_index
    dg_full_edges = dg_nt.edge_index
    dgt_full_edges = dgt_nt.edge_index
    full_edges = [ppi_full_edges, dd_full_edges, dg_full_edges, dgt_full_edges]

    x = [ppi_nt.x, dd_nt.x, dg_nt.x, dgt_nt.x]

    method = "gcn"
    if method == "gcn":
        model = GCN(all_nt, epoch)
    elif method == "vgae":
        model = run_vgae(ppi_nt, dd_nt, dg_nt, dgt_nt, epoch)

    z_full_edges = model.encode(x, full_edges)

    pp_pred = np.dot(z_full_edges[0].detach().numpy(), z_full_edges[0].detach().numpy().T)
    dd_pred = np.dot(z_full_edges[1].detach().numpy(), z_full_edges[1].detach().numpy().T)
    dg_pred = np.dot(z_full_edges[2].detach().numpy(), z_full_edges[2].detach().numpy().T)
    dg_pred[:ppi_nt.num_nodes, :ppi_nt.num_nodes] = pp_pred
    num_all_nodes = ppi_nt.num_nodes + dd_nt.num_nodes
    dg_pred[ppi_nt.num_nodes:num_all_nodes, ppi_nt.num_nodes:num_all_nodes] = dd_pred

    # Min-Max Normalization
    adj_pred = (dg_pred - dg_pred.min()) / (dg_pred.max() - dg_pred.min())

    true_adj_matrix = nx.to_numpy_array(all_nt).astype(int)

    # Define function to calculate accuracy based on threshold
    def calculate_accuracy(threshold):
        z_adj = (adj_pred >= threshold).astype(int)
        accuracy = np.sum(z_adj == true_adj_matrix)
        return accuracy

    thresholds = np.linspace(0, 1, 10)
    accuracies = [calculate_accuracy(th) for th in thresholds]

    # Find max accuracy and the corresponding threshold
    max_accuracy = max(accuracies)
    best_threshold = thresholds[np.argmax(accuracies)]

    # Plot accuracy vs threshold
    plt.plot(thresholds, accuracies)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.grid(True)
    plt.show()

    print(max_accuracy, best_threshold)
