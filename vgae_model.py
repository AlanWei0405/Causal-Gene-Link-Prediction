import torch
from torch_geometric.nn import VGAE, GCNConv


class VGAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.dd_conv1 = GCNConv(in_channels, 2 * out_channels)
        # self.dg_conv1 = GCNConv(in_channels, 2 * out_channels)
        # self.dg_conv1 = GCNConv(in_channels, 2 * out_channels)
        self.gg_conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)

        # self.dd_conv_mu = GCNConv(2 * out_channels, out_channels)
        # self.dg_conv_mu = GCNConv(2 * out_channels, out_channels)
        # self.dg_conv_mu = GCNConv(2 * out_channels, out_channels)
        self.gg_conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)

        # self.dd_conv_logstd = GCNConv(2 * out_channels, out_channels)
        # self.dg_conv_logstd = GCNConv(2 * out_channels, out_channels)
        # self.dg_conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.gg_conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = torch.relu(self.gg_conv1(x, edge_index))
        return self.gg_conv_mu(x, edge_index), self.gg_conv_logstd(x, edge_index)
