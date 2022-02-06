import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import degree
from torch_sparse import spmm


class GCN(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(GCN, self).__init__()
        self.W = nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(self, x, h, t):
        """
        x: matrix of shape Exd
        h: vector of length T or matrix of shape Tx1
        t: vector of length T or matrix of shape Tx1

        PlainGCN(X) = sigma(D^(1/2) M D^(1/2) X)
        sigma is activation function
        M is adjacency matrix of shape ExE
        D is degree matrix of shape ExE
        X is embedding matrix of shape Exd
        """
        deg = degree(h, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[t] * deg_inv_sqrt[h]

        # x = F.selu(spmm(torch.cat([t, h], dim= 0), norm, x.size(0), x.size(0), self.W(x)))
        x = F.relu(spmm(torch.cat([t, h], dim=0), norm, x.size(0), x.size(0), self.W(x)))
        return x


class PlainGCN(nn.Module):
    def __init__(self):
        super(PlainGCN, self).__init__()

    def forward(self, x, h, t):
        """
        x: matrix of shape Exd
        h: vector of length T or matrix of shape Tx1
        t: vector of length T or matrix of shape Tx1

        PlainGCN(X) = sigma(D^(1/2) M D^(1/2) X)
        sigma is activation function
        M is adjacency matrix of shape ExE
        D is degree matrix of shape ExE
        X is embedding matrix of shape Exd
        """
        deg = degree(h, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[t] * deg_inv_sqrt[h]

        # x = F.selu(spmm(torch.cat([t, h], dim= 0), norm, x.size(0), x.size(0), x))
        x = F.relu(spmm(torch.cat([t, h], dim=0), norm, x.size(0), x.size(0), x))
        return x
