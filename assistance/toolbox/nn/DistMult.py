import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreDistMult(nn.Module):
    def __init__(self, input_dropout_rate=0.2):
        super(CoreDistMult, self).__init__()
        self.dropout = nn.Dropout(input_dropout_rate)

    def forward(self, h, r):
        h = self.dropout(h)
        r = self.dropout(r)

        x = h * r
        x = F.relu(x)
        return x


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout_rate=0.2):
        super(DistMult, self).__init__()
        self.E = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.R = nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.core = CoreDistMult(input_dropout_rate)
        self.loss = nn.BCELoss()
        self.b = nn.Parameter(torch.zeros(num_entities))

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.E(h_idx)
        r = self.R(r_idx)

        t = self.core(h, r)
        t = t.view(-1, self.embedding_dim)

        x = torch.mm(t, self.E.weight.transpose(1, 0))
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x
