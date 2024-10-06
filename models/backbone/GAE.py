import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
import numpy as np
import scipy as sp
import torch.nn.functional as F

class GAEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GAEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),

        )
        self.gat = GCNConv(channel // reduction, channel, bias=False)
        self.s = nn.Sigmoid()

    def create_edge_index(self, adj, device):
        adj = adj.cpu()
        ones = torch.ones_like(adj)
        zeros = torch.zeros_like(adj)
        edge_index = torch.where(adj > 0, ones, zeros)
        #
        edge_index_temp = sp.sparse.coo_matrix(edge_index.numpy())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(indices)
        # edge_weight
        edge_weight = []
        t = edge_index.numpy().tolist()
        for x, y in zip(t[0], t[1]):
            edge_weight.append(adj[x, y])
        edge_weight = torch.FloatTensor(edge_weight)
        edge_weight = edge_weight.unsqueeze(1)

        return edge_index.to(device), edge_weight.to(device)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = torch.unsqueeze(y, 1)
        s = torch.randn((y.shape[0], y.shape[1], c)).to(x.device)
        for k in range(y.shape[0]):
            feat = y[k, :, :]  # s, h
            # creat edge_index
            adj = torch.matmul(feat, feat.T)  # s * s
            adj = F.softmax(adj, dim=1)
            edge_index, edge_weight = self.create_edge_index(adj, x.device)
            feat = self.gat(feat, edge_index, edge_weight)
            s[k, :, :] = feat
        s = s[:, -1, :]
        y = torch.as_tensor(s.view(b, c, 1, 1), dtype=x.dtype)
        return x * y.expand_as(x)

