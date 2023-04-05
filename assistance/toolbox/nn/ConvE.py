import torch
import torch.nn.functional as F
from torch import nn


class CoreConvE(nn.Module):
    def __init__(self, embedding_dim, img_h=10, input_dropout=0.2, hidden_dropout1=0.3, hidden_dropout2=0.2):
        super(CoreConvE, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.feature_map_drop = nn.Dropout2d(hidden_dropout1)
        self.hidden_drop = nn.Dropout(hidden_dropout2)

        self.img_h = img_h
        self.img_w = embedding_dim // self.img_h

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        hidden_size = (self.img_h * 2 - 3 + 1) * (self.img_w - 3 + 1) * 32
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, h, r):
        h = h.view(-1, 1, self.img_h, self.img_w)
        r = r.view(-1, 1, self.img_h, self.img_w)

        x = torch.cat([h, r], 2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, hidden_dropout=0.3):
        super(ConvE, self).__init__()
        self.E = nn.Embedding(num_entities, embedding_dim)
        self.R = nn.Embedding(num_relations, embedding_dim)

        self.core = CoreConvE(embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout)
        self.b = nn.Parameter(torch.zeros(num_entities))
        self.m = nn.PReLU()

        self.loss = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, h, r):
        h = self.E(h)  # Bxd
        r = self.R(r)  # Bxd
        t = self.core(h, r)

        x = torch.mm(t, self.dropout(self.E.weight).transpose(1, 0))
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x  # batch_size x E
