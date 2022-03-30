import numpy as np
import torch
import torch.nn as nn


class TuckERTNT(nn.Module):
    def __init__(self, d, de, dr, dt, device="cpu", input_dropout=0., hidden_dropout1=0., hidden_dropout2=0., **kwargs):
        super(TuckERTNT, self).__init__()

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
        self.E = nn.Embedding(self.ne, de)
        self.R = nn.Embedding(self.nr, dr)
        self.T = nn.Embedding(self.nt, dt)

        # Core tensor
        self.W = nn.Parameter(torch.tensor(np.random.uniform(-0.1, 0.1, (dr, de, dt, de)), dtype=torch.float, device=self.device, requires_grad=True))

        # "Special" Layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)
        self.loss = nn.BCELoss()

        self.bne = nn.BatchNorm1d(de)

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.T.weight.data)

    def forward(self, e1_idx, r_idx, t_idx):
        ### Temporal part
        # Mode 1 product with entity vector
        e1 = self.E(e1_idx)
        x = self.bne(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.de)  # (B, 1, de)

        # Mode 2 product with relation vector
        r = self.R(r_idx)  # (B, dr)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # (B, dr) * (dr, de*de*dt) = (B, de*de*dt)
        W_mat = W_mat.view(-1, self.de, self.de * self.dt)  # (B, de, de*dt)
        x = torch.bmm(x, W_mat)  # (B, 1, de) * (B, de, de*dt) = (B, 1, de*dt)

        # Mode 4 product with entity matrix 
        x = x.view(-1, self.de)  # (B, de*dt) -> (B*dt, de)
        x = torch.mm(x, self.E.weight.transpose(1, 0))  # (B*dt, de) * (E, de)^T = (B*dt, E)

        # Mode 3 product with time vector
        t = self.T(t_idx).view(-1, 1, self.dt)  # (B, 1, dt)
        xt = x.view(-1, self.dt, self.ne)  # (B, dt, E)
        xt = torch.bmm(t, xt)  # (B, 1, dt) * (B, dt, E) -> (B, 1, E)
        xt = xt.view(-1, self.ne)  # (B, E)

        ### Non temporal part
        # mode 3 product with identity matrix
        x = x.view(-1, self.dt)  # (B*E, dt)
        x = torch.mm(x, torch.ones(self.dt).to(self.device).view(self.dt, 1))  # (B*E, dt) * (dt, 1) = (B*E, 1)
        x = x.view(-1, self.ne)  # (B, E)

        # Sum of the 2 models
        x = x + xt

        # Turn results into "probabilities"
        pred = torch.sigmoid(x)
        return pred
