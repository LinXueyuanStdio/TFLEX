import numpy as np
import torch
import torch.nn as nn


class TuckERTTT(nn.Module):
    def __init__(self, d, de, dr, dt, ranks, device='cpu', input_dropout=0., hidden_dropout1=0., hidden_dropout2=0., **kwargs):
        super(TuckERTTT, self).__init__()

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

        ## Core Tensor
        # Size of Tensor Ring decompostion tensors
        ni = [self.dr, self.de, self.de, self.dt]
        if isinstance(ranks, int) or isinstance(ranks, np.int64):
            ranks = [ranks for _ in range(3)]
        elif isinstance(ranks, list) and len(ranks) == 3:
            pass
        else:
            raise TypeError('ranks must be int or list of len 3')

        list_tmp = [1]
        list_tmp.extend(ranks)
        list_tmp.append(1)
        ranks = list_tmp

        # List of tensors of the tensor train 
        self.Glist = nn.ParameterList([
            nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks[i], ni[i], ranks[i + 1])), dtype=torch.float, requires_grad=True).to(self.device))
            for i in range(4)
        ])

        # "Special" Layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)
        self.loss = nn.BCELoss()

        self.bne = nn.BatchNorm1d(de)
        self.bnr = nn.BatchNorm1d(dr)
        self.bnt = nn.BatchNorm1d(dt)

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx, t_idx):

        e1 = self.E(e1_idx)
        r = self.R(r_idx)
        t = self.T(t_idx)

        # Product between embedding matrices and TT-cores
        # RG1 = torch.mm(r,self.Glist[0].view(-1,self.dr))
        # G2E = torch.einsum('aib,ic->acb',(self.Glist[1],e1))
        # G3E = torch.einsum('aib,ic->acb',(self.Glist[2],self.E))
        # TG4 = torch.mm(t,self.Glist[3].view(-1,self.dr).transpose())

        W = torch.einsum('i1,1j2,3k4,4l->ijkl', list(self.Glist))
        W = W.view(self.dr, self.de, self.de, self.dt)

        # Mode 1 product with entity vector
        x = e1
        x = x.view(-1, 1, self.de)

        # Mode 2 product with relation vector
        W_mat = torch.mm(r, W.view(self.dr, -1))
        W_mat = W_mat.view(-1, self.de, self.de * self.dt)
        x = torch.bmm(x, W_mat)

        # Mode 3 product with temporal vector
        t = self.T(t_idx)
        x = x.view(-1, self.de, self.dt)
        x = torch.bmm(x, t.view(*t.shape, -1))

        # Mode 4 product with entity matrix
        x = x.view(-1, self.de)
        x = torch.mm(x, self.E.weight.transpose(1, 0))

        # Turn results into "probabilities"
        pred = torch.sigmoid(x)
        return pred
