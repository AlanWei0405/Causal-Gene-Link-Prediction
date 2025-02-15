import numpy as np
import torch
from sklearn.decomposition import PCA


def combine_features(disease_sum, gene_sum, train_data):

    gene_bio_embed = np.stack(gene_sum['Embedding'].to_list())
    disease_bio_embed = np.stack(disease_sum['Embedding'].to_list())

    # Using PCA to reduce dimension
    pca = PCA(n_components=128)
    gene_bio_embed = pca.fit_transform(gene_bio_embed)
    disease_bio_embed = pca.fit_transform(disease_bio_embed)
    additional_gene_features = torch.tensor(gene_bio_embed, dtype=train_data['gene'].x.dtype)
    additional_disease_features = torch.tensor(disease_bio_embed, dtype=train_data['disease'].x.dtype)

    # Concatenate topological and biobert features along the feature dimension
    train_data['gene'].x = torch.cat([train_data['gene'].x, additional_gene_features], dim=1)
    train_data['disease'].x = torch.cat([train_data['disease'].x, additional_disease_features], dim=1)

    return train_data

def reducing_training_edges(train_data, val_data, test_data, leave_out_ratio):

    undirected_edge_types = [('gene', 'interact', 'gene'), ('disease', 'relate', 'disease')]
    directed_edge_pairs = [(('gene', 'associate', 'disease'), ('disease', 'rev_associate', 'gene'))]

    """
    for undirected edges, if we remove (123, 456) in one edge type, 
        we should also remove (456, 123) in the same edge type;
    for directed edges, if we remove (321, 654) in ('gene', 'associate', 'disease'), 
        we should also remove (654, 321) in ('disease', 'rev_associate', 'gene').
    """

    for edge_type in undirected_edge_types:
        edge_index = train_data[edge_type].edge_index

        # Split edges
        edge_index_keep, edge_index_leave_out = cut_edges(edge_index, leave_out_ratio)

        # Update train/val/test sets
        train_data[edge_type].edge_index = edge_index_keep

        # Convert to [num_edges, 2] format for efficient comparison
        all_edges = edge_index.T  # Shape [num_edges, 2]
        removed_edges = edge_index_leave_out.T  # Shape [num_removed_edges, 2]

        # Identify which edges should be removed
        mask = ~torch.isin(all_edges, removed_edges).all(dim=1)

        # Apply mask to keep only edges that are NOT in edge_index_leave_out
        val_data[edge_type].edge_index = edge_index[:, mask]
        test_data[edge_type].edge_index = edge_index[:, mask]

    for edge_type, rev_edge_type in directed_edge_pairs:
        edge_index = train_data[edge_type].edge_index
        rev_edge_index = train_data[rev_edge_type].edge_index

        # Split edges
        edge_index_keep, edge_index_leave_out = cut_edges(edge_index, leave_out_ratio)

        # Ensure reverse edges are also removed
        rev_edge_index_keep = torch.flip(edge_index_keep, [0])  # Flip (reverse) edges
        rev_edge_index_leave_out = torch.flip(edge_index_leave_out, [0])

        # Update train/val/test sets
        train_data[edge_type].edge_index = edge_index_keep
        train_data[rev_edge_type].edge_index = rev_edge_index_keep

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

        return train_data, val_data, test_data

def cut_edges(edge_index, leave_out_ratio, seed=45):

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

    # Remove both directions of selected edges
    leave_out_mask = torch.isin(inverse_indices, leave_out_idx)
    keep_mask = ~leave_out_mask  # Keep the remaining edges

    # Get the edges to keep/cut
    edge_index_keep = edge_index[:, keep_mask]
    edge_index_leave_out = edge_index[:, leave_out_mask]

    return edge_index_keep, edge_index_leave_out
