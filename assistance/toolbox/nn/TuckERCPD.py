import numpy as np
import torch
import torch.nn as nn


class TuckERCPD(torch.nn.Module):
    def __init__(self, d, de, dr, dt, device="cpu", **kwargs):
        super(TuckERCPD, self).__init__()

        self.device = device

        # Embeddings dimensionality
        self.de = de
        self.dr = dr
        self.dt = dt

        # Data dimensionality
        self.ne = len(d.entities)
        self.nr = len(d.relations)
        self.nt = len(d.time)

        # CPD rank
        self.p = min(self.de, self.dt, self.dr)

        # Embedding matrices
        self.E = torch.nn.Embedding(self.ne, de).to(self.device)
        self.R = torch.nn.Embedding(self.nr, dr).to(self.device)
        self.T = torch.nn.Embedding(self.nt, dt).to(self.device)

        ### CPD Decomp of core tensor of Tucker
        # Core identity tensor of CPD
        self.I = torch.zeros(*[self.p for _ in range(4)]).to(self.device)
        for i in range(self.p):
            self.I[i, i, i, i] = 1

        # Factors of CPD
        self.Flist = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (d, self.p)), dtype=torch.float, requires_grad=True).to(self.device))
            for d in [self.dr, self.de, self.de, self.dt]
        ])

        # Dropout Layers
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.hidden_dropout3 = torch.nn.Dropout(kwargs["hidden_dropout3"])

        # batchnorm layers
        self.bne = torch.nn.BatchNorm1d(self.de)
        self.bnp3 = torch.nn.BatchNorm1d(self.p ** 3)
        self.bnp2 = torch.nn.BatchNorm1d(self.p ** 2)
        self.bnp = torch.nn.BatchNorm1d(self.p)

        # Loss
        self.loss = torch.nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.T.weight.data)

        for i in range(len(self.Flist)):
            nn.init.xavier_normal_(self.Flist[i])

    def forward(self, e1_idx, r_idx, t_idx):

        # Select corresponding embeddings
        e1 = self.E(e1_idx)
        e1 = self.bne(e1)
        e1 = self.input_dropout(e1)

        r = self.R(r_idx)
        t = self.T(t_idx)

        # Compute intermediate factor matrices from embeddings and factor of core tensor

        fr = torch.mm(r, self.Flist[0])
        fe1 = torch.mm(e1, self.Flist[1])
        FE = torch.mm(self.E.weight, self.Flist[2])
        ft = torch.mm(t, self.Flist[3])

        ### Recover tensor

        # Mode 1 product with intermediate relaton vecot
        x = torch.mm(fr, self.I.view(self.p, -1))
        x = self.bnp3(x)
        x = self.hidden_dropout1(x)

        x = x.view(-1, self.p, self.p ** 2)
        x = torch.bmm(fe1.view(-1, 1, self.p), x).view(-1, self.p ** 2)
        x = self.bnp2(x)
        x = self.hidden_dropout2(x)

        x = x.view(-1, self.p, self.p)
        x = torch.bmm(x, ft.view(*ft.shape, -1)).view(-1, self.p)
        x = self.bnp(x)
        x = self.hidden_dropout3(x)

        x = torch.mm(x, FE.transpose(1, 0))

        # Turn results into "probabilities"
        pred = torch.sigmoid(x)
        return pred
