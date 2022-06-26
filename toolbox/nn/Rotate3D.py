"""
@date: 2021/10/30
@description: null
"""
import torch
from torch import nn

pi = 3.14159262358979323846

class CoreRotate3D(nn.Module):
    def __init__(self, entity_dim):
        super(CoreRotate3D, self).__init__()
        self.entity_dim = entity_dim

    def forward(self, h, r):
        h = h.view(-1, 1, self.entity_dim)
        r = r.view(-1, self.entity_dim, self.entity_dim)

        t = torch.bmm(h, r).view(-1, self.entity_dim)
        return t

        head_i, head_j, head_k = torch.chunk(head, 3, dim=2)
        beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=2)
        tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=2)

        bias = torch.abs(bias)

        # Make phases of relations uniformly distributed in [-pi, pi]
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Obtain representation of the rotation axis
        rel_i = torch.cos(beta_1)
        rel_j = torch.sin(beta_1)*torch.cos(beta_2)
        rel_k = torch.sin(beta_1)*torch.sin(beta_2)

        C = rel_i*head_i + rel_j*head_j + rel_k*head_k
        C = C*(1-cos_theta)

        # Rotate the head entity
        new_head_i = head_i*cos_theta + C*rel_i + sin_theta*(rel_j*head_k-head_j*rel_k)
        new_head_j = head_j*cos_theta + C*rel_j - sin_theta*(rel_i*head_k-head_i*rel_k)
        new_head_k = head_k*cos_theta + C*rel_k + sin_theta*(rel_i*head_j-head_i*rel_j)

        score_i = new_head_i*bias - tail_i
        score_j = new_head_j*bias - tail_j
        score_k = new_head_k*bias - tail_k

        score = torch.stack([score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score

class Rotate3D(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, input_dropout=0.3):
        super(Rotate3D, self).__init__()
        self.entity_dim = entity_dim

        self.E = nn.Embedding(num_entities, entity_dim)
        self.R = nn.Embedding(num_relations, entity_dim * entity_dim)

        self.core = CoreRotate3D(entity_dim)
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
