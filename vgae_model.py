from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.nn.models.autoencoder import EPS, MAX_LOGSTD
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score


class VGAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.9):
        super().__init__()

        self.num_graphs = 4  # Number of graphs
        self.convs = nn.ModuleList([GCNConv(in_channels, 2 * out_channels, cached=True)
                                    for _ in range(self.num_graphs)])
        self.convs_mu = nn.ModuleList([GCNConv(2 * out_channels, out_channels, cached=True)
                                       for _ in range(self.num_graphs)])
        self.convs_logstd = nn.ModuleList([GCNConv(2 * out_channels, out_channels, cached=True)
                                           for _ in range(self.num_graphs)])
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x, edge_index):
        mu = []
        logstd = []
        for i in range(self.num_graphs):
            x_i = self.convs[i](x[i], edge_index[i]).relu()
            x_i_act = self.dropout(x_i)
            mu_i = self.convs_mu[i](x_i_act, edge_index[i])
            mu_i_act = self.dropout(mu_i)
            logstd_i = self.convs_logstd[i](x_i_act, edge_index[i])
            logstd_i_act = self.dropout(logstd_i)
            mu.append(mu_i_act)
            logstd.append(logstd_i_act)
        return mu, logstd


class MultiVGAE(VGAE):
    def __init__(self, encoder, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)
        self.__mu__ = None
        self.__logstd__ = None

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = [logstd.clamp(max=MAX_LOGSTD) for logstd in self.__logstd__]
        z = [self.reparametrize(mu_i, logstd_i) for mu_i, logstd_i in zip(self.__mu__, self.__logstd__)]
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else [logstd_i.clamp(max=MAX_LOGSTD) for logstd_i in logstd]

        total_kl_loss = []
        for mu_i, logstd_i in zip(mu, logstd):
            kl = -0.5 * torch.mean(torch.sum(1 + 2 * logstd_i - mu_i ** 2 - logstd_i.exp() ** 2, dim=1))
            total_kl_loss.append(kl)
        return total_kl_loss

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        total_recon_loss = 0
        for i, z_i in enumerate(z):
            p_i = pos_edge_index[i]
            if neg_edge_index is None:
                n_i = negative_sampling(p_i, z_i.size(0))
            else:
                n_i = neg_edge_index[i]

            pos_loss = -torch.log(self.decoder(z_i, p_i, sigmoid=True) + EPS).mean()
            neg_loss = -torch.log(1 - self.decoder(z_i, n_i, sigmoid=True) + EPS).mean()
            total_recon_loss += (pos_loss + neg_loss)
        return total_recon_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        auc = []
        ap = []
        for i, z_i in enumerate(z):
            pos_y = z_i.new_ones(pos_edge_index[i].size(1))
            neg_y = z_i.new_zeros(neg_edge_index[i].size(1))
            y = torch.cat([pos_y, neg_y], dim=0)

            pos_pred = self.decoder(z_i, pos_edge_index[i], sigmoid=True)
            neg_pred = self.decoder(z_i, neg_edge_index[i], sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            auc.append(roc_auc_score(y, pred))
            ap.append(average_precision_score(y, pred))

        return auc, ap
