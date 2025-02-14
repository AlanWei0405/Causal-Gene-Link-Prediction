import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.nn import to_hetero

from models.heterogeneous.gat_encoder import GATEncoder
from models.heterogeneous.graphconv_encoder import GraphConvEncoder
from models.heterogeneous.sage_encoder import SAGEEncoder
from models.heterogeneous.hetero_gae import HeteroGAE
from utils.meta2vec import get_features
from utils.split_data import split_data
from sklearn.decomposition import PCA


def run_onheterograph(hetero_data, gene_sum, disease_sum, epoch):

    hetero_data, test_data, train_data, val_data = split_data(hetero_data)

    # Process each undirected edge type
    undirected_edge_types = [('gene', 'interact', 'gene'), ('disease', 'relate', 'disease')]
    for edge_type in undirected_edge_types:
        edge_index = train_data[edge_type].edge_index

        # Split edges
        edge_index_keep, edge_index_leave_out = cut_edges(edge_index)

        # Update train/val/test sets
        train_data[edge_type].edge_index = edge_index_keep

        # --- Optimized Edge Removal for Val/Test ---
        # Convert to [num_edges, 2] format for efficient comparison
        all_edges = edge_index.T  # Shape [num_edges, 2]
        removed_edges = edge_index_leave_out.T  # Shape [num_removed_edges, 2]

        # Identify which edges should be removed
        mask = ~torch.isin(all_edges, removed_edges).all(dim=1)

        # Apply mask to keep only edges that are NOT in edge_index_leave_out
        val_data[edge_type].edge_index = edge_index[:, mask]
        test_data[edge_type].edge_index = edge_index[:, mask]  # Same for test

    # Process gene-disease edges and their reverse
    directed_edge_pairs = [
        (('gene', 'associate', 'disease'), ('disease', 'rev_associate', 'gene'))
    ]

    for edge_type, rev_edge_type in directed_edge_pairs:
        edge_index = train_data[edge_type].edge_index
        rev_edge_index = train_data[rev_edge_type].edge_index

        # Split edges
        edge_index_keep, edge_index_leave_out = cut_edges(edge_index)

        # Ensure reverse edges are also removed
        rev_edge_index_keep = torch.flip(edge_index_keep, [0])  # Flip (reverse) edges
        rev_edge_index_leave_out = torch.flip(edge_index_leave_out, [0])

        # Update train/val/test sets
        train_data[edge_type].edge_index = edge_index_keep
        train_data[rev_edge_type].edge_index = rev_edge_index_keep

        # --- Optimized Edge Removal for Val/Test ---
        # Create a 2-row concatenated tensor to compare efficiently
        all_edges = edge_index.T  # Shape [num_edges, 2]
        removed_edges = edge_index_leave_out.T  # Shape [num_removed_edges, 2]

        # Identify which edges should be removed
        mask = ~torch.isin(all_edges, removed_edges).all(dim=1)

        # Apply mask to keep only edges that are NOT in edge_index_leave_out
        val_data[edge_type].edge_index = edge_index[:, mask]
        val_data[rev_edge_type].edge_index = rev_edge_index[:, mask]
        test_data[edge_type].edge_index = edge_index[:, mask]
        test_data[rev_edge_type].edge_index = rev_edge_index[:, mask]

    train_data = get_features(train_data)

    gene_bio_embed = np.stack(gene_sum['Embedding'].to_list())
    disease_bio_embed = np.stack(disease_sum['Embedding'].to_list())

    # Using PCA to reduce dimension
    pca = PCA(n_components=128)
    gene_bio_embed = pca.fit_transform(gene_bio_embed)
    disease_bio_embed = pca.fit_transform(disease_bio_embed)

    additional_gene_features = torch.tensor(gene_bio_embed, dtype=train_data['gene'].x.dtype)
    additional_disease_features = torch.tensor(disease_bio_embed, dtype=train_data['disease'].x.dtype)

    # # Check dimensions
    # assert additional_gene_features.size(0) == train_data['gene'].x.size(0), "Mismatch in number of gene nodes"
    # assert additional_disease_features.size(0) == train_data['disease'].x.size(0), "Mismatch in number of disease nodes"

    # Concatenate features along the feature dimension
    train_data['gene'].x = torch.cat([train_data['gene'].x, additional_gene_features], dim=1)
    train_data['disease'].x = torch.cat([train_data['disease'].x, additional_disease_features], dim=1)

    # print("Updated gene features shape:", train_data['gene'].x.shape)
    # print("Updated disease features shape:", train_data['disease'].x.shape)

    val_data['gene'].x = train_data['gene'].x
    test_data['gene'].x = train_data['gene'].x
    val_data['disease'].x = train_data['disease'].x
    test_data['disease'].x = train_data['disease'].x


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Choose the encoder for VGAE
    # encoder = to_hetero(GraphConvEncoder(64, 32), hetero_data.metadata(), 'sum')
    encoder = to_hetero(SAGEEncoder(64, 32), hetero_data.metadata(), 'sum')
    # encoder = to_hetero(GATEncoder(64, 32), hetero_data.metadata(), 'sum')

    model = HeteroGAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(train_data.x_dict, train_data.edge_index_dict)

    train_losses=[]
    val_losses=[]
    auc_list=[]
    ap_list=[]

    with tqdm(range(epoch), unit="epoch") as epoch_bar:
        epoch_bar.set_description("training loop")
        for i in epoch_bar:
            model.train()
            optimizer.zero_grad()
            out = model.encode(train_data.x_dict, train_data.edge_index_dict)
            loss = model.recon_loss(out, train_data[('gene', 'associate', 'disease')].pos_edge_label_index,
                                    train_data[('gene', 'associate', 'disease')].neg_edge_label_index)
            # loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().numpy())

            model.eval()
            with torch.no_grad():  # No need to calculate gradients for validation
                z = model.encode(val_data.x_dict, val_data.edge_index_dict)
                val_loss = model.recon_loss(z, val_data[('gene', 'associate', 'disease')].pos_edge_label_index,
                                            val_data[('gene', 'associate', 'disease')].neg_edge_label_index)
                # val_loss = val_loss + (1 / val_data.num_nodes) * model.kl_loss()
                val_losses.append(val_loss)

            if (i + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    z = model.encode(test_data.x_dict, test_data.edge_index_dict)
                    auc, ap, f1_max, best_threshold = model.test(z, test_data[('gene', 'associate', 'disease')].pos_edge_label_index,
                                            test_data[('gene', 'associate', 'disease')].neg_edge_label_index)
                    top_at_k_test = model.test_top_k(z, test_data[('gene', 'associate', 'disease')].edge_index,
                                                     test_data[('gene', 'associate', 'disease')].pos_edge_label_index,
                                                     k=1908)
                    print(auc, ap, f1_max)
                    print(top_at_k_test)
                    auc_list.append(auc)
                    ap_list.append(ap)

    # Plot AUC and AP values
    plt.figure(figsize=(10, 6))
    plt.plot(auc_list, label=f'AUC', linestyle='-')
    plt.plot(ap_list, label=f'AP', linestyle='--')
    plt.xlabel('Epochs (per 20 epochs)')
    plt.ylabel('Score')
    plt.title('AUC and AP over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    highest_auc = max(auc_list)
    highest_ap = max(ap_list)

    print(f"Highest AUC achieved: {highest_auc:.4f}")
    print(f"Highest AP achieved: {highest_ap:.4f}")

    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()

def cut_edges(edge_index, leave_out_ratio=0.9, seed=45):

    torch.manual_seed(seed)

    # Sort node pairs to identify unique edges (Since our input is an undirected graph,
    # removing one edge requires operations on both directions)
    sorted_edges, _ = torch.sort(edge_index, dim=0)  # Sort nodes in each edge
    unique_edges, inverse_indices = torch.unique(sorted_edges, dim=1, return_inverse=True)

    num_edges = unique_edges.size(1)
    num_leave_out = int(num_edges * leave_out_ratio)

    # Randomly select unique edges to leave out
    perm = torch.randperm(num_edges)
    leave_out_idx = perm[:num_leave_out]  # Unique edges to remove
    keep_idx = perm[num_leave_out:]  # Unique edges to keep

    # Remove both directions of selected edges
    leave_out_mask = torch.isin(inverse_indices, leave_out_idx)
    keep_mask = ~leave_out_mask  # Keep the remaining edges

    # Get the edges to keep/cut
    edge_index_keep = edge_index[:, keep_mask]
    edge_index_leave_out = edge_index[:, leave_out_mask]

    return edge_index_keep, edge_index_leave_out
