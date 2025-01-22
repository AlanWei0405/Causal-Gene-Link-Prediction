import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr='max', normalize=True)
        self.conv2 = SAGEConv((-1, -1), out_channels, aggr='max', normalize=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
