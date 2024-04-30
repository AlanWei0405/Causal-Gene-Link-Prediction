from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import negative_sampling


class VGAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.num_graphs = 4  # Number of graphs
        self.convs = nn.ModuleList([GCNConv(in_channels, 2 * out_channels, cached=True)
                                    for _ in range(self.num_graphs)])
        self.convs_mu = nn.ModuleList([GCNConv(2 * out_channels, out_channels, cached=True)
                                       for _ in range(self.num_graphs)])
        self.convs_logstd = nn.ModuleList([GCNConv(2 * out_channels, out_channels, cached=True)
                                           for _ in range(self.num_graphs)])

        # self.gg_conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        # self.dg_conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        # self.dgt_conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        # self.dd_conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        #
        # self.gg_conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dg_conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dgt_conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dd_conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        #
        # self.gg_conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dg_conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dgt_conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.dd_conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x1, edge_index1, x2, edge_index2, x3, edge_index3, x4, edge_index4):
        x = [x1, x2, x3, x4]
        edge_index = [edge_index1, edge_index2, edge_index3, edge_index4]
        mu = []
        logstd = []
        for i in range(self.num_graphs):
            x_i = self.convs[i](x[i], edge_index[i])
            mu_i = self.convs_mu[i](x_i)
            logstd_i = self.convs_logstd[i](x_i)
            mu.append(mu_i)
            logstd.append(logstd_i)
        return mu, logstd


class MultiVGAE(VGAE):
    def __init__(self, encoder, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)
        self.__mu__ = None
        self.__logstd__ = None

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        z = [self.reparametrize(mu_i, logstd_i) for mu_i, logstd_i in zip(self.__mu__, self.__logstd__)]
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else [logstd_i.clamp(max=1.0) for logstd_i in logstd]

        total_loss = 0
        for mu_i, logstd_i in zip(mu, logstd):
            kl = -0.5 * torch.mean(torch.sum(1 + 2 * logstd_i - mu_i ** 2 - logstd_i.exp() ** 2, dim=1))
            total_loss += kl
        return total_loss

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):

        total_recon_loss = 0
        for i, z in enumerate(z):
            pos_edge_index = pos_edge_index[i]
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            else:
                neg_edge_index = neg_edge_index[i]

            pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-10).mean()
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-10).mean()
            total_recon_loss += pos_loss + neg_loss

        return total_recon_loss
