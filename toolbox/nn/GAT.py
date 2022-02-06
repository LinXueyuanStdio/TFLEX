import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax
from torch_sparse import spmm


class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, h, t):
        """
        x: matrix of shape Exd
        h: vector of length T or matrix of shape Tx1
        t: vector of length T or matrix of shape Tx1

        GAT(X) = AX
        A is self attention from X
        A is matrix of shape ExE
        X is embedding matrix of shape Exd
        """
        e_i = self.a_i(x)[h].view(-1)
        e_j = self.a_j(x)[t].view(-1)
        e = e_i + e_j
        alpha = softmax(F.leaky_relu(e).float(), h)
        sparse_index = torch.cat([h.view(1, -1), t.view(1, -1)], dim=0)
        x = F.relu(spmm(sparse_index, alpha, x.size(0), x.size(0), x))
        return x
