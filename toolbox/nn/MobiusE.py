import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexScoringAll, ComplexBatchNorm1d, ComplexMult, ComplexAdd, ComplexDiv, ComplexAlign
from toolbox.nn.MobiusEmbedding import MobiusEmbedding, MobiusDropout, MobiusBatchNorm1d
from toolbox.nn.Regularizer import N3


def mobius_mul_with_unit_norm(Q_1, Q_2):
    a_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}

    # Normalize the relation to eliminate the scaling effect
    denominator = torch.sqrt(a_r ** 2 + b_r ** 2 + c_r ** 2 + d_r ** 2)
    p = a_r / denominator
    q = b_r / denominator
    u = c_r / denominator
    v = d_r / denominator
    #  Q'=E Hamilton product R
    h_r = (a_h * p + q) / (a_h * u + v)
    return h_r


def mobius_mul(Q_1, Q_2):
    a_h = Q_1  # = {a_h : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}
    h_r = (a_h * a_r + b_r) / (a_h * c_r + d_r)
    return h_r


class MobiusMult(nn.Module):
    """
            a_r * h + b_r
    h * r = -------------
            c_r * h + d_r
    h in CP^d
    r_a, r_b, r_c, r_d in C^d
    """

    def __init__(self, norm_flag=False):
        super(MobiusMult, self).__init__()
        self.flag_hamilton_mul_norm = norm_flag
        self.complex_mul = ComplexMult(norm_flag)
        self.complex_add = ComplexAdd()
        self.complex_div = ComplexDiv()

    def forward(self, h, r):
        r_a, r_b, r_c, r_d = r
        hr_top = self.complex_add(self.complex_mul(h, r_a), r_b)
        hr_bottom = self.complex_add(self.complex_mul(h, r_c), r_d)
        h_r = self.complex_div(hr_top, hr_bottom)
        return h_r


class MobiusE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3, regularization_weight=0.1):
        super(MobiusE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = nn.BCELoss()
        self.flag_hamilton_mul_norm = norm_flag
        self.E = ComplexEmbedding(self.num_entities, self.embedding_dim, 2)  # a + bi
        self.R = MobiusEmbedding(self.num_relations, self.embedding_dim, 4)  # 4 numbers: a + bi
        self.E_dropout = ComplexDropout([input_dropout, input_dropout])
        self.R_dropout = MobiusDropout([[input_dropout, input_dropout]] * 4)
        self.hidden_dp = ComplexDropout([hidden_dropout, hidden_dropout])
        self.E_bn = ComplexBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = MobiusBatchNorm1d(self.embedding_dim, 4)
        self.b = nn.Parameter(torch.zeros(num_entities))

        self.mul = MobiusMult(norm_flag)
        self.scoring_all = ComplexScoringAll()
        self.align = ComplexAlign()
        self.regularizer = N3(regularization_weight)

    def forward(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        return self.forward_head_batch(h_idx, r_idx)

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
        x = score_a + score_b
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)

        return x

    def regular_loss(self, h_idx, r_idx):
        h = self.E(h_idx)
        r = self.R(r_idx)
        h_a, h_a_i = h
        (r_a, r_a_i), (r_b, r_b_i), (r_c, r_c_i), (r_d, r_d_i) = r
        factors = (
            torch.sqrt(h_a ** 2 + h_a_i ** 2),
            torch.sqrt(r_a ** 2 + r_a_i ** 2 + r_c ** 2 + r_c_i ** 2 + r_b ** 2 + r_b_i ** 2 + r_d ** 2 + r_d_i ** 2),
        )
        regular_loss = self.regularizer(factors)
        return regular_loss

    def reverse_loss(self, h_idx, r_idx, max_relation_idx):
        h = self.E(h_idx)
        h_a, h_b = h
        h = (h_a.detach(), h_b.detach())

        r = self.R(r_idx)
        reverse_rel_idx = (r_idx + max_relation_idx) % (2 * max_relation_idx)

        t = self.mul(h, r)
        reverse_r = self.R(reverse_rel_idx)
        reverse_t = self.mul(t, reverse_r)
        reverse_a, reverse_b = self.align(reverse_t, h)  # a + b i
        reverse_score = reverse_a + reverse_b
        reverse_score = torch.mean(F.relu(reverse_score))

        return reverse_score

    def init(self):
        self.E.init()
        self.R.init()

#
#          TRAIN:  {'MRR': 0.3367987126111984, 'hits@[1,3,10]': tensor([0.2347, 0.3808, 0.5365])}
#          VALID :  {'MRR': 0.28631188720464706, 'hits@[1,3,10]': tensor([0.2089, 0.3107, 0.4430])}
#
#
# TEST :  ({'rhs': 0.38042595982551575, 'lhs': 0.1826966404914856}, {'rhs': tensor([0.2927, 0.4182, 0.5544]), 'lhs': tensor([0.1148, 0.1959, 0.3186])})
