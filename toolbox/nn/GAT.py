import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax
from torch_sparse import spmm


class GAT(nn.Module):
    """
    第一种GAT，可以不用关系，只用实体对
    """

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


class GAT2(nn.Module):
    def __init__(self, hidden_dim):
        super(GAT2, self).__init__()
        self.a_i = nn.Linear(hidden_dim, 1, bias=False)
        self.a_j = nn.Linear(hidden_dim, 1, bias=False)
        self.a_k = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, E, R, T):
        """
        E: 矩阵 |E| x d_e ，即实体数 x 嵌入维度
        R: 矩阵 |R| x d_r ，即关系数 x 嵌入维度
        T: 矩阵 |T| x 3， 即三元组数 x 3，每一行是[头实体索引，关系索引，尾实体索引]
        """
        h = T[:, 0]
        r = T[:, 1]
        t = T[:, 2]
        e_i = self.a_i(E)[h].view(-1)
        e_j = self.a_j(E)[t].view(-1)
        r_k = self.a_j(R)[r].view(-1)
        e = e_i + r_k + e_j
        alpha = softmax(F.leaky_relu(e).float(), h)
        sparse_index = torch.cat([h.view(1, -1), t.view(1, -1)], dim=0)
        E = F.relu(spmm(sparse_index, alpha, E.size(0), E.size(0), E))
        return E
