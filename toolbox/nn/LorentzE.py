"""
@date: 2022/3/17
@description: null
"""
import torch
import torch.nn as nn


class LorentzE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout_rate=0.2):
        super(LorentzE, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        self.E_ct = nn.Embedding(num_entities, embedding_dim)
        self.E_x = nn.Embedding(num_entities, embedding_dim)
        self.E_y = nn.Embedding(num_entities, embedding_dim)
        self.E_z = nn.Embedding(num_entities, embedding_dim)  # E x d

        self.R_v1 = nn.Embedding(num_relations, embedding_dim)  # R x d
        self.R_v2 = nn.Embedding(num_relations, embedding_dim)
        self.R_v3 = nn.Embedding(num_relations, embedding_dim)

        self.dropout = nn.Dropout(input_dropout_rate)
        self.loss = nn.BCELoss()

    def forward(self, h_idx, r_idx):
        # h_idx Bx1
        # r_idx Bx1
        h_ct = self.E_ct(h_idx).view(-1, self.embedding_dim)  # Bxd
        h_x = self.E_x(h_idx).view(-1, self.embedding_dim)  # Bxd
        h_y = self.E_y(h_idx).view(-1, self.embedding_dim)  # Bxd
        h_z = self.E_z(h_idx).view(-1, self.embedding_dim)  # Bxd
        r_v1 = self.R_v1(h_idx).view(-1, self.embedding_dim)  # Bxd
        r_v2 = self.R_v2(h_idx).view(-1, self.embedding_dim)  # Bxd
        r_v3 = self.R_v3(h_idx).view(-1, self.embedding_dim)  # Bxd
        # f : x -> [0, c]
        t_ct = 1 * h_ct + r_v1 * h_x + r_v2 * h_y + r_v3 * h_z
        t_x = h_x
        t_y = h_y
        t_z = h_z
        # 1 vs. N , N=E
        score_ct = torch.mm(t_ct, self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4
        score = score.sigmoid()
        return score
