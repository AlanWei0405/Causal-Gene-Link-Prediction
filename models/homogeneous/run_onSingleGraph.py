import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.nn import GAE, VGAE
from models.homogeneous.GCN.gcn_encoder import GCNEncoder
from models.homogeneous.GCN.vgcn_encoder import VariationalGCNEncoder
from utils.meta2vec import get_features
from utils.split_data import split_data


def run_onsinglegraph(hetero_data, epoch):

    hetero_data, test_data, train_data, val_data = split_data(hetero_data, 'single graph')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = GCNEncoder(128, 32)
    # encoder = VariationalGCNEncoder(128, 32)
    model = GAE(encoder).to(device)
    # model = VGAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_losses=[]
    val_losses=[]
    auc_list=[]
    ap_list=[]

    with tqdm(range(epoch), unit="epoch") as epoch_bar:
        epoch_bar.set_description("training loop")
        for i in epoch_bar:
            model.train()
            optimizer.zero_grad()
            out = model.encode(train_data.x, train_data.edge_index)
            loss = model.recon_loss(out, train_data.pos_edge_label_index,
                                    train_data.neg_edge_label_index)
            # loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().numpy())

            model.eval()
            with torch.no_grad():  # No need to calculate gradients for validation
                z = model.encode(val_data.x, val_data.edge_index)
                val_loss = model.recon_loss(z, val_data.pos_edge_label_index,
                                            val_data.neg_edge_label_index)
                # val_loss = val_loss + (1 / val_data.num_nodes) * model.kl_loss()
                val_losses.append(val_loss)

            if i % 20 == 0:
                model.eval()
                with torch.no_grad():
                    z = model.encode(test_data.x, test_data.edge_index)
                    auc, ap = model.test(z, test_data.pos_edge_label_index,
                                            test_data.neg_edge_label_index)
                    print(auc, ap)
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
