import time

import numpy as np
import torch
from vgae_model import VGAEModel, MultiVGAE
from torch_geometric.utils import train_test_split_edges


def run_vgae(ppi_data, dd_data, dg_data, dgt_data, epoch):

    ppi_full_edges = ppi_data.edge_index
    dd_full_edges = dd_data.edge_index
    dg_full_edges = dg_data.edge_index
    dgt_full_edges = dgt_data.edge_index
    full_edges = [ppi_full_edges, dd_full_edges, dg_full_edges, dgt_full_edges]

    # Initialize
    ppi_data = train_test_split_edges(ppi_data)
    dd_data = train_test_split_edges(dd_data)
    dg_data = train_test_split_edges(dg_data)
    dgt_data = train_test_split_edges(dgt_data)

    in_channels = ppi_data.num_node_features
    out_channels = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiVGAE(VGAEModel(in_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = [ppi_data.x, dd_data.x, dg_data.x, dgt_data.x]
    train_pos_index = [ppi_data.train_pos_edge_index, dd_data.train_pos_edge_index,
                       dg_data.train_pos_edge_index, dgt_data.train_pos_edge_index]
    test_pos_index = [ppi_data.test_pos_edge_index, dd_data.test_pos_edge_index,
                      dg_data.test_pos_edge_index, dgt_data.test_pos_edge_index]
    test_neg_index = [ppi_data.test_neg_edge_index, dd_data.test_neg_edge_index,
                      dg_data.test_neg_edge_index, dgt_data.test_neg_edge_index]
    coef = np.array([1 / ppi_data.num_nodes, 1 / dd_data.num_nodes, 1 / dg_data.num_nodes, 1 / dgt_data.num_nodes])

    start_time = time.time()
    for i in range(1, epoch + 1):
        epoch_start_time = time.time()
        model, loss = train(model, optimizer, x, train_pos_index, coef)
        print(f'Epoch {i}, Loss: {loss:.4f}')
        epoch_end_time = time.time()
        print(f"Epoch {i} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        if i % 1 == 0:
            names = ['PP', 'DD', 'DG', 'DGT']
            auc, ap = test(model, x, train_pos_index, test_pos_index, test_neg_index)
            for j in range(len(auc)):
                print('Epoch: {:03d}, Name: {}, AUC: {:.4f}, AP: {:.4f}'.format(i, names[j], auc[j], ap[j]))
    end_time = time.time()  # After the end of the training loop
    print(f"Total training time: {end_time - start_time:.2f} seconds.")

    z = model.encode(x, full_edges)
    z1 = torch.cat([z[1], z[2]], dim=0)
    z2 = torch.cat([z[0], z[3]], dim=0)
    adj = torch.matmul(z1, z2.t())
    return adj


def train(model, optimizer, x, train_pos_index, coef):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_index)
    loss = model.recon_loss(z, train_pos_index)
    loss = loss + sum([coef[i] * model.kl_loss()[i] for i in range(len(coef))])
    loss.backward()
    optimizer.step()
    return model, loss.item()


def test(model, x, train_pos_index, test_pos_index, test_neg_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_index)
        return model.test(z, test_pos_index, test_neg_index)
