import numpy as np
import torch
from torch import nn


class CoreTuckER(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dropout1=0.4, hidden_dropout2=0.5):
        super(CoreTuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.W = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (relation_dim, entity_dim, entity_dim))))

        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        self.bn0 = nn.BatchNorm1d(entity_dim)
        self.bn1 = nn.BatchNorm1d(entity_dim)

        self.m = nn.PReLU()

    def forward(self, h, r):
        h = self.bn0(h.view(-1, self.entity_dim)).view(-1, 1, self.entity_dim)

        W = self.W.view(self.relation_dim, -1)
        W = torch.mm(r.view(-1, self.relation_dim), W)
        W = W.view(-1, self.entity_dim, self.entity_dim)
        W = self.hidden_dropout1(W)

        t = torch.bmm(h, W)
        t = t.view(-1, self.entity_dim)
        t = self.bn1(t)
        t = self.hidden_dropout2(t)
        t = self.m(t)
        return t

    def w(self, h, r):
        h = torch.cat([h.transpose(1, 0).unsqueeze(dim=0)] * r.size(0), dim=0)  # BxdxE

        W = self.W.view(self.relation_dim, -1)
        W = torch.mm(r.view(-1, self.relation_dim), W)
        W = W.view(-1, self.entity_dim, self.entity_dim)  # Bxdxd
        W = self.hidden_dropout1(W)
        t = torch.bmm(W, h)  # BxdxE
        return t


class TuckER(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim, input_dropout=0.3, hidden_dropout=0.3, hidden_dropout2=0.3):
        super(TuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.E = nn.Embedding(num_entities, entity_dim)
        self.R = nn.Embedding(num_relations, relation_dim)

        self.core = CoreTuckER(entity_dim, relation_dim, hidden_dropout, hidden_dropout2)
        self.input_dropout = nn.Dropout(input_dropout)

        self.loss = nn.BCELoss()
        self.b = nn.Parameter(torch.zeros(num_entities))

    def init(self):
        nn.init.kaiming_uniform_(self.E.weight.data)
        nn.init.kaiming_uniform_(self.R.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.input_dropout(self.E(h_idx))
        r = self.R(r_idx)

        t = self.core(h, r)
        t = t.view(-1, self.entity_dim)

        x = torch.mm(t, self.input_dropout(self.E.weight).transpose(1, 0))
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x
