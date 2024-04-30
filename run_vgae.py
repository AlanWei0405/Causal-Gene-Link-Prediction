import torch
from vgae_model import VGAEModel, MultiVGAE
from torch_geometric.utils import train_test_split_edges


def run_vgae(ppi_data, dd_data, dg_data, epoch):

    # Initialize
    ppi_data = train_test_split_edges(ppi_data)
    dd_data = train_test_split_edges(dd_data)
    dg_data = train_test_split_edges(dg_data)

    in_channels = ppi_data.num_node_features
    out_channels = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiVGAE(VGAEModel(in_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(1, epoch + 1):
        model, loss = train(model, optimizer, ppi_data)
        print(f'Epoch {i}, Loss: {loss:.4f}')
        if i % 10 == 0:
            auc, ap = test(model, ppi_data)
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(i, auc, ap))


def train(model, optimizer, ppi_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(ppi_data.x, ppi_data.train_pos_edge_index)
    loss = model.recon_loss(z, ppi_data.train_pos_edge_index)
    loss = loss + (1 / ppi_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return model, loss.item()


def test(model, ppi_data):
    model.eval()
    with torch.no_grad():
        z = model.encode(ppi_data.x, ppi_data.train_pos_edge_index)
        return model.test(z, ppi_data.test_pos_edge_index, ppi_data.test_neg_edge_index)
