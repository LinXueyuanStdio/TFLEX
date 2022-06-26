"""
@date: 2021/10/30
@description: null
"""
import torch
from torch import nn


class CoreRESCAL(nn.Module):
    def __init__(self, entity_dim):
        super(CoreRESCAL, self).__init__()
        self.entity_dim = entity_dim

    def forward(self, h, r):
        h = h.view(-1, 1, self.entity_dim)
        r = r.view(-1, self.entity_dim, self.entity_dim)

        t = torch.bmm(h, r).view(-1, self.entity_dim)
        return t


class RESCAL(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, input_dropout=0.3):
        super(RESCAL, self).__init__()
        self.entity_dim = entity_dim

        self.E = nn.Embedding(num_entities, entity_dim)
        self.R = nn.Embedding(num_relations, entity_dim * entity_dim)

        self.core = CoreRESCAL(entity_dim)
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
