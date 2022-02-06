import torch
import torch.nn.functional as F
from torch import nn


class CoreParamEGate(nn.Module):
    def __init__(self, entity_dim, hidden_dim=100):
        super(CoreParamEGate, self).__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, entity_dim)

    def forward(self, h, r1, r2):
        h = h.view(-1, 1, self.entity_dim)
        r1 = r1.view(-1, self.entity_dim, self.hidden_dim)
        r2 = r2.view(-1, self.entity_dim, self.hidden_dim)
        gate = torch.sigmoid(torch.bmm(h, r1).squeeze(dim=1))
        value = torch.tanh(torch.bmm(h, r2).squeeze(dim=1))
        x = (1 - gate) * value
        x = self.linear(x)
        x = F.relu(x)
        return x


class CoreParamECNN(nn.Module):
    def __init__(self, entity_dim,
                 conv_num_channels=32,
                 conv_num_channels2=64,
                 conv_filter_height=3, conv_filter_width=3,
                 hidden_dropout1: float = 0.2,
                 hidden_dropout2: float = 0.2):
        super(CoreParamECNN, self).__init__()
        self.entity_dim = entity_dim

        self.conv_in_height = 10
        self.conv_in_width = self.entity_dim // 10

        self.conv_filter_height = conv_filter_height
        self.conv_filter_width = conv_filter_width
        self.conv_num_channels = conv_num_channels
        self.conv_num_channels2 = conv_num_channels2

        self.conv_out_height = self.conv_in_height - self.conv_filter_height + 1
        self.conv_out_width = self.conv_in_width - self.conv_filter_width + 1

        self.conv_out_height2 = self.conv_out_height - self.conv_filter_height + 1
        self.conv_out_width2 = self.conv_out_width - self.conv_filter_width + 1

        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        hidden_dim = self.conv_num_channels * self.conv_num_channels2 * self.conv_out_height2 * self.conv_out_width2
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, entity_dim)

    def forward(self, h, r1, r2):
        img = h.view(1, -1, self.conv_in_height, self.conv_in_width)
        batch_size = img.size(1)
        conv_weight1 = r1.view(batch_size * self.conv_num_channels, 1, self.conv_filter_height, self.conv_filter_width)
        conv_weight2 = r2.view(batch_size * self.conv_num_channels2, 1, self.conv_filter_height, self.conv_filter_width)

        x = F.conv2d(img, weight=conv_weight1, groups=batch_size)
        x = F.relu(x)
        x = self.hidden_dropout1(x)
        x = x.view(-1, batch_size, self.conv_out_height, self.conv_out_width)

        x = F.conv2d(x, weight=conv_weight2, groups=batch_size)
        x = F.relu(x)
        x = self.hidden_dropout2(x)
        x = x.view(batch_size, self.hidden_dim)

        x = self.linear(x)
        x = F.relu(x)
        return x


class ParamE(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim, hidden_dropout=0.2):
        super(ParamE, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.E = nn.Embedding(num_entities, entity_dim)

        conv_num_channels = 32
        conv_num_channels2 = 64
        conv_filter_height = 3
        conv_filter_width = 3
        self.R1 = nn.Embedding(num_relations, conv_num_channels * conv_filter_height * conv_filter_width)
        self.R2 = nn.Embedding(num_relations, conv_num_channels2 * conv_filter_height * conv_filter_width)

        self.core = CoreParamECNN(entity_dim,
                                  conv_num_channels, conv_num_channels2,
                                  conv_filter_height, conv_filter_width)
        self.dropout = nn.Dropout(hidden_dropout)
        self.b = nn.Parameter(torch.zeros(num_entities))
        self.m = nn.PReLU()

        self.loss = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R1.weight.data)
        nn.init.xavier_normal_(self.R2.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.E(h_idx)  # Bxd
        r1 = self.R1(r_idx)  # Bxd
        r2 = self.R2(r_idx)  # Bxd

        t = self.core(h, r1, r2)
        t = t.view(-1, self.entity_dim)

        x = torch.mm(t, self.dropout(self.E.weight).transpose(1, 0))
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x  # batch_size x E
