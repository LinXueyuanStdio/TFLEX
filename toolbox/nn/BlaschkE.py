import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexScoringAll, ComplexBatchNorm1d, ComplexMult, ComplexAdd, ComplexDiv


class BlaschkeMult(nn.Module):
    """
                     h - r
    h * r = --------------
            r_conj * h - 1
    h in C^d
    r in C^d
    """

    def __init__(self, norm_flag=False):
        super(BlaschkeMult, self).__init__()
        self.flag_hamilton_mul_norm = norm_flag
        self.complex_mul = ComplexMult(norm_flag)
        self.complex_add = ComplexAdd()
        self.complex_div = ComplexDiv()

    def forward(self, h, r):
        r_a, r_b = r
        # h_a, h_b = h
        # r = F.normalize(torch.cat([r_a.unsqueeze(dim=1), r_b.unsqueeze(dim=1)], dim=1), dim=1, p=2)
        # r_a = r[:, 0, :]
        # r_b = r[:, 1, :]
        # h = F.normalize(torch.cat([h_a.unsqueeze(dim=1), h_b.unsqueeze(dim=1)], dim=1), dim=1, p=2)
        # h_a = h[:, 0, :]
        # h_b = h[:, 1, :]
        # print(r_a.size(), r_b.size(), h_a.size(), h_b.size())
        # r_norm = torch.sqrt(r_a ** 2 + r_b ** 2)
        # r_a = r_a / r_norm
        # r_b = r_b / r_norm
        # h_norm = torch.sqrt(h_a ** 2 + h_b ** 2)
        # h_a = h_a / h_norm
        # h_b = h_b / h_norm
        # h = (h_a, h_b)

        neg_r = (-r_a, -r_b)
        hr_top = self.complex_add(h, neg_r)

        neg_one = (-torch.ones_like(r_a), torch.zeros_like(r_b))  # -1 = -1 + 0 i = (-1, 0)
        conjugate_r = (r_a, -r_b)
        hr_bottom = self.complex_add(self.complex_mul(h, conjugate_r), neg_one)

        h_r = self.complex_div(hr_top, hr_bottom)
        return h_r


class BlaschkE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3):
        super(BlaschkE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = nn.BCELoss()
        self.flag_hamilton_mul_norm = norm_flag
        self.E = ComplexEmbedding(self.num_entities, self.embedding_dim, 2)  # a + bi
        self.R = ComplexEmbedding(self.num_relations, self.embedding_dim, 2)  # a + bi
        self.E_dropout = ComplexDropout([input_dropout, input_dropout])
        self.R_dropout = ComplexDropout([input_dropout, input_dropout])
        self.hidden_dp = ComplexDropout([hidden_dropout, hidden_dropout])
        self.E_bn = ComplexBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = ComplexBatchNorm1d(self.embedding_dim, 4)

        self.mul = BlaschkeMult(norm_flag)
        self.scoring_all = ComplexScoringAll()

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

        t = self.mul(h, r)
        if self.flag_hamilton_mul_norm:
            score_a, score_b = self.scoring_all(t, self.E.get_embeddings())  # a + b i
        else:
            score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(self.E_bn(self.E.get_embeddings())))
        score = score_a + score_b
        return torch.sigmoid(score)

    def init(self):
        self.E.init()
        self.R.init()
