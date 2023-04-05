import numpy as np
import torch
from torch import nn

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexScoringAll, ComplexBatchNorm1d


class CoreTuckER(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dropout1=0.4):
        super(CoreTuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.W = nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (relation_dim, entity_dim, entity_dim))))

        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)

    def forward(self, h, r):
        h = h.view(-1, 1, self.entity_dim)

        W = self.W.view(self.relation_dim, -1)
        W = torch.mm(r.view(-1, self.relation_dim), W)
        W = W.view(-1, self.entity_dim, self.entity_dim)
        W = self.hidden_dropout1(W)

        t = torch.bmm(h, W)
        t = t.view(-1, self.entity_dim)
        return t

    def w(self, h, r):
        h = torch.cat([h.transpose(1, 0).unsqueeze(dim=0)] * r.size(0), dim=0)  # BxdxE

        W = self.W.view(self.relation_dim, -1)
        W = torch.mm(r.view(-1, self.relation_dim), W)
        W = W.view(-1, self.entity_dim, self.entity_dim)  # Bxdxd
        W = self.hidden_dropout1(W)
        t = torch.bmm(W, h)  # BxdxE
        return t


class ComplexTuckER(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dropout=0.4):
        super(ComplexTuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.Wa = CoreTuckER(entity_dim, relation_dim, hidden_dropout)
        self.Wb = CoreTuckER(entity_dim, relation_dim, hidden_dropout)

        self.bn0 = nn.BatchNorm1d(entity_dim)
        self.bn1 = nn.BatchNorm1d(entity_dim)
        self.bn3 = nn.BatchNorm1d(relation_dim)
        self.bn4 = nn.BatchNorm1d(relation_dim)
        self.bn5 = nn.BatchNorm1d(entity_dim)
        self.bn6 = nn.BatchNorm1d(entity_dim)
        self.ma = nn.PReLU()
        self.mb = nn.PReLU()

    def forward(self, h, r):
        h_a, h_b = h
        h_a = self.bn0(h_a)
        h_b = self.bn1(h_b)
        r_a, r_b = r
        r_a = self.bn3(r_a)
        r_b = self.bn4(r_b)
        t_a = self.Wa(h_a, r_a) - self.Wb(h_a, r_b) - self.Wb(h_b, r_a) - self.Wa(h_b, r_b)
        t_b = self.Wb(h_a, r_a) + self.Wa(h_a, r_b) + self.Wa(h_b, r_a) - self.Wb(h_b, r_b)
        t_a = self.bn5(t_a)
        t_b = self.bn6(t_b)
        t_a = self.ma(t_a)
        t_b = self.mb(t_b)
        return t_a, t_b

class TuckER(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim, input_dropout=0.3, hidden_dropout=0.3, hidden_dropout2=0.3):
        super(TuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.flag_hamilton_mul_norm = True

        self.E = ComplexEmbedding(num_entities, entity_dim)
        self.R = ComplexEmbedding(num_relations, relation_dim)

        self.core = ComplexTuckER(entity_dim, relation_dim, hidden_dropout)
        self.E_bn = ComplexBatchNorm1d(entity_dim, 2)
        self.E_dropout = ComplexDropout([input_dropout, input_dropout])
        self.R_dropout = ComplexDropout([input_dropout, input_dropout])
        self.hidden_dp = ComplexDropout([hidden_dropout, hidden_dropout])

        self.scoring_all = ComplexScoringAll()
        self.bce = nn.BCELoss()
        self.b1 = nn.Parameter(torch.zeros(num_entities))
        self.b2 = nn.Parameter(torch.zeros(num_entities))

    def init(self):
        self.E.init()
        self.R.init()

    def forward(self, h_idx, r_idx):
        return self.forward_head_batch(h_idx.view(-1), r_idx.view(-1))

    def forward_head_batch(self, h_idx, r_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(h_idx)
        r = self.R(r_idx)

        t = self.core(h, r)

        if self.flag_hamilton_mul_norm:
            score_a, score_b = self.scoring_all(t, self.E.get_embeddings())  # a + b i
        else:
            score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(self.E_bn(self.E.get_embeddings())))
        score_a = score_a + self.b1.expand_as(score_a)
        score_b = score_b + self.b2.expand_as(score_b)

        y_a = torch.sigmoid(score_a)
        y_b = torch.sigmoid(score_b)

        return y_a, y_b

    def loss(self, target, y):
        y_a, y_b = target
        return self.bce(y_a, y) + self.bce(y_b, y)
