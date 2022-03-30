import numpy as np
import torch
import torch.nn as nn


class TuckERTTR(nn.Module):
    def __init__(self, d, de, dr, dt, ranks, device='cpu', input_dropout=0., hidden_dropout1=0., hidden_dropout2=0., **kwargs):
        super(TuckERTTR, self).__init__()

        self.device = device

        # Embeddings dimensionality
        self.de = de
        self.dr = dr
        self.dt = dt

        # Data dimensionality
        self.ne = len(d.entities)
        self.nr = len(d.relations)
        self.nt = len(d.time)

        # Embedding matrices
        self.E = nn.Embedding(self.ne, de).to(self.device)
        self.R = nn.Embedding(self.nr, dr).to(self.device)
        self.T = nn.Embedding(self.nt, dt).to(self.device)

        # Size of Tensor Ring decompostion tensors
        ni = [self.dr, self.de, self.de, self.dt]
        if isinstance(ranks, int) or isinstance(ranks, np.int64):
            ranks = [ranks for _ in range(5)]
        elif isinstance(ranks, list) and len(ranks) == 5:
            pass
        else:
            raise TypeError('ranks must be int or list of len 5')

        # List of tensors of the TR
        self.Zlist = nn.ParameterList([
            nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks[i], ni[i], ranks[i + 1])), dtype=torch.float, requires_grad=True).to(self.device))
            for i in range(4)
        ])

        # dropout Layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        # batchnorm layers
        self.bne = nn.BatchNorm1d(de)

        # loss
        self.loss = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx, t_idx):

        e1 = self.E(e1_idx)
        r = self.R(r_idx)
        t = self.T(t_idx)

        # Recover core tensor from TR (compute the trace of hadamart (element wise) product of all tensors)
        W = torch.einsum('aib,bjc,ckd,dla->ijkl', list(self.Zlist))
        W = W.view(self.dr, self.de, self.de, self.dt)

        # Mode 1 product with entity vector
        x = e1
        x = x.view(-1, 1, self.de)

        # Mode 2 product with relation vector
        W_mat = torch.mm(r, W.view(self.dr, -1))
        W_mat = W_mat.view(-1, self.de, self.de * self.dt)
        x = torch.bmm(x, W_mat)

        # Mode 3 product with temporal vector
        x = x.view(-1, self.de, self.dt)
        x = torch.bmm(x, t.view(*t.shape, -1))

        # Mode 4 product with entity matrix
        x = x.view(-1, self.de)
        x = torch.mm(x, self.E.weight.transpose(1, 0))

        # Turn results into "probabilities"
        pred = torch.sigmoid(x)
        return pred
