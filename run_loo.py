import random
import os
import numpy as np
from matplotlib import pyplot as plt

from utils.build_networks import build_networks
from run_vgae import run_vgae


def run_loo(ppi_data, dd_data, dg_data, epoch, reps = 1):
    ranks=[]
    for r in range(reps):
        # Randomly remove and save an edge
        ppi_copy = ppi_data.copy()
        dd_copy = dd_data.copy()
        dg_copy = dg_data.copy()
        random_index = random.randint(0, dg_copy.shape[0] - 1)
        removed_edge = dg_copy.iloc[random_index, :].copy()
        dg_copy = dg_copy.drop(random_index)

        # Build networks with manipulated data
        ppi_nt, dd_nt, dg_nt, dgt_nt, all_nt = build_networks(ppi_copy, dd_copy, dg_copy)

        ppi_full_edges = ppi_nt.edge_index
        dd_full_edges = dd_nt.edge_index
        dg_full_edges = dg_nt.edge_index
        dgt_full_edges = dgt_nt.edge_index
        full_edges = [ppi_full_edges, dd_full_edges, dg_full_edges, dgt_full_edges]

        x = [ppi_nt.x, dd_nt.x, dg_nt.x, dgt_nt.x]

        model, seen_edges = run_vgae(ppi_nt, dd_nt, dg_nt, dgt_nt, epoch)

        z_loo = model.encode(x, full_edges)

        pp_pred_loo = np.dot(z_loo[0].detach().numpy(), z_loo[0].detach().numpy().T)
        dd_pred_loo = np.dot(z_loo[1].detach().numpy(), z_loo[1].detach().numpy().T)
        dg_pred_loo = np.dot(z_loo[2].detach().numpy(), z_loo[2].detach().numpy().T)

        dg_pred_loo[:ppi_nt.num_nodes, :ppi_nt.num_nodes] = pp_pred_loo
        num_all_nodes = ppi_nt.num_nodes + dd_nt.num_nodes
        dg_pred_loo[ppi_nt.num_nodes:num_all_nodes, ppi_nt.num_nodes:num_all_nodes] = dd_pred_loo
        dg_pred_loo_norm = (dg_pred_loo - dg_pred_loo.min()) / (dg_pred_loo.max() - dg_pred_loo.min())

        gene_node, disease_node = removed_edge
        row_scores = dg_pred_loo_norm[disease_node, :ppi_nt.num_nodes]

        seen_edges = set(tuple(edge.tolist()) for edge in seen_edges)
        unseen_indices = [j for j in range(num_all_nodes) if (disease_node, j) not in seen_edges]

        # Step 2: Get the predicted scores for the unseen entries
        unseen_scores = dg_pred_loo_norm[disease_node, unseen_indices]

        sorted_indices = np.argsort(unseen_scores)[::-1]
        # sorted_scores = row_scores[sorted_indices]
        rank = np.where(np.isin(sorted_indices, gene_node))[0] + 1  # 1-based rank
        ranks.append(rank)

        # Specify the path to your .pth file
        model_path = "model.pth"
        ppi_nt_path = 'ppi_nt.pkl'
        dd_nt_path = 'dd_nt.pkl'
        dg_nt_path = 'dg_nt.pkl'
        dgt_nt_path = 'dgt_nt.pkl'
        all_nt_path = 'all_nt.pkl'

        # Check if the file exists before trying to delete it
        if os.path.exists(model_path):
            os.remove(model_path)
            os.remove(ppi_nt_path)
            os.remove(dd_nt_path)
            os.remove(dg_nt_path)
            os.remove(dgt_nt_path)
            os.remove(all_nt_path)
            print("File deleted successfully.")
        else:
            print("File not found.")

    plt.plot(ranks)
    plt.xlabel("Test")
    plt.ylabel("Rank")
    plt.title("LOO Test Ranks")
    plt.grid(True)
    plt.show()

        # print(f"Rank of the predicted score for the removed edge ({disease_node}, {gene_node}) "
        #       f"among all possible edges connected to node {disease_node}: {rank}")

        # top_k = 10
        # top_k_edges = sorted_indices[:top_k]
        # top_k_scores = node1_scores[top_k_edges]
        #
        # print(f"Top-{top_k} predicted edges for node {disease_node}:")
        # for i in range(top_k):
        #     print(f"Edge ({disease_node}, {top_k_edges[i]}): Score = {top_k_scores[i]}")
