import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from vgae_model import VGAEModel, MultiVGAE
from torch_geometric.utils import train_test_split_edges


def run_vgae(ppi_data, dd_data, dg_data, dgt_data, epoch):

    model_path = 'model.pth'

    # ppi_full_edges = ppi_data.edge_index
    # dd_full_edges = dd_data.edge_index
    # dg_full_edges = dg_data.edge_index
    # dgt_full_edges = dgt_data.edge_index
    # full_edges = [ppi_full_edges, dd_full_edges, dg_full_edges, dgt_full_edges]

    in_channels = ppi_data.num_node_features
    out_channels = 4

    model = MultiVGAE(VGAEModel(in_channels, out_channels))

    # x = [ppi_data.x, dd_data.x, dg_data.x, dgt_data.x]

    if os.path.exists(model_path):
        model = MultiVGAE(VGAEModel(in_channels, out_channels))
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Split data
        ppi_data = train_test_split_edges(ppi_data)
        dd_data = train_test_split_edges(dd_data)
        dg_data = train_test_split_edges(dg_data)
        dgt_data = train_test_split_edges(dgt_data)

        x = [ppi_data.x, dd_data.x, dg_data.x, dgt_data.x]

        train_pos_index = [ppi_data.train_pos_edge_index, dd_data.train_pos_edge_index,
                           dg_data.train_pos_edge_index, dgt_data.train_pos_edge_index]
        val_pos_index = [ppi_data.val_pos_edge_index, dd_data.val_pos_edge_index,
                         dg_data.val_pos_edge_index, dgt_data.val_pos_edge_index]
        test_pos_index = [ppi_data.test_pos_edge_index, dd_data.test_pos_edge_index,
                          dg_data.test_pos_edge_index, dgt_data.test_pos_edge_index]
        test_neg_index = [ppi_data.test_neg_edge_index, dd_data.test_neg_edge_index,
                          dg_data.test_neg_edge_index, dgt_data.test_neg_edge_index]
        # Weight 1
        coef = np.array([1 / ppi_data.num_nodes, 1 / dd_data.num_nodes, 1 / dg_data.num_nodes, 1 / dgt_data.num_nodes])
        # # Weight 2
        # coef = np.array([ppi_data.num_edges, dd_data.num_edges, dg_data.num_edges, dgt_data.num_edges])

        train_losses = []
        val_losses = []

        auc_values = {name: [] for name in ['PP', 'DD', 'DG', 'DGT']}
        ap_values = {name: [] for name in ['PP', 'DD', 'DG', 'DGT']}

        # start_time = time.time()
        with tqdm(range(epoch), unit="epoch") as epoch_bar:
            epoch_bar.set_description("training loop")
            for i in epoch_bar:
                epoch_start_time = time.time()
                model, loss = train(model, optimizer, x, train_pos_index, coef)
                # print(f'Epoch {i}, Loss: {loss:.4f}')
                train_losses.append(loss)
                # epoch_end_time = time.time()
                # print(f"Epoch {i} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

                # Validation phase
                val_loss = validate(model, x, val_pos_index, coef)
                val_losses.append(val_loss)

                if i % 10 == 0:
                    print("\n")
                    names = ['PP', 'DD', 'DG', 'DGT']
                    auc, ap = test(model, x, test_pos_index, test_neg_index)

                    # Store AUC and AP values
                    for j, name in enumerate(names):
                        auc_values[name].append(auc[j])
                        ap_values[name].append(ap[j])
                        # print('Epoch: {:03d}, Name: {}, AUC: {:.4f}, AP: {:.4f}'.format(i, name, auc[j], ap[j]))

            # Plot AUC and AP values
            plt.figure(figsize=(10, 6))

            # Define line styles for different names
            colors = ['blue', 'green', 'red', 'purple']
            for i, name in enumerate(names):
                plt.plot(auc_values[name], label=f'{name} AUC', color=colors[i], linestyle='-')
                plt.plot(ap_values[name], label=f'{name} AP', color=colors[i], linestyle='--')

            plt.xlabel('Epochs (per 10 epochs)')
            plt.ylabel('Score')
            plt.title('AUC and AP over Training')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Plot training and validation loss
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.show()

        # end_time = time.time()  # After the end of the training loop
        # print(f"Total training time: {end_time - start_time:.2f} seconds.")
        torch.save(model.state_dict(), 'model.pth')

        # model = MultiVGAE(VGAEModel(in_channels, out_channels))
        # model.load_state_dict(torch.load('model.pth'))
        # model.eval()
        #
        # # Compare the state dictionaries of both models
        # model_dict = model.state_dict()
        # loaded_model_dict = model.state_dict()
        #
        # are_models_equal = True
        # for key in model_dict:
        #     if not torch.equal(model_dict[key], loaded_model_dict[key]):
        #         print(f"Mismatch found at {key}")
        #         are_models_equal = False
        #
        # if are_models_equal:
        #     print("The saved and loaded models are identical.")
        # else:
        #     print("The saved and loaded models are NOT identical.")

        print()

    # z = model.encode(x, full_edges)
    #
    # pp_pred = np.dot(z[0].detach().numpy(), z[0].detach().numpy().T)
    # dd_pred = np.dot(z[1].detach().numpy(), z[1].detach().numpy().T)
    # dg_pred = np.dot(z[2].detach().numpy(), z[2].detach().numpy().T)
    #
    # dg_pred[:ppi_data.num_nodes, :ppi_data.num_nodes] = pp_pred
    # num_all_nodes = ppi_data.num_nodes + dd_data.num_nodes
    # dg_pred[ppi_data.num_nodes:num_all_nodes, ppi_data.num_nodes:num_all_nodes] = dd_pred
    # dg_pred1 = (dg_pred - dg_pred.min()) / (dg_pred.max() - dg_pred.min())  # max-min normalization

    return model, dg_data.train_pos_edge_index


def train(model, optimizer, x, train_pos_index, coef, beta=0.5):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_index)
    loss = model.recon_loss(z, train_pos_index)
    loss = beta * loss + (1 - beta) * sum([coef[i] * model.kl_loss()[i] for i in range(len(coef))])
    loss.backward()
    optimizer.step()
    return model, loss.item()

def validate(model, x, val_pos_index, coef, beta=0.5):
    model.eval()
    with torch.no_grad():  # No need to calculate gradients for validation
        z = model.encode(x, val_pos_index)
        loss = model.recon_loss(z, val_pos_index)
        loss = beta * loss + (1 - beta) * sum([coef[i] * model.kl_loss()[i] for i in range(len(coef))])
    return loss

def test(model, x, test_pos_index, test_neg_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, test_pos_index)
        return model.test(z, test_pos_index, test_neg_index)
