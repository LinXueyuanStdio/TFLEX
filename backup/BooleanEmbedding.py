from typing import List, Tuple

import torch
from torch import nn

from toolbox.nn.ComplexEmbedding import (
    ComplexEmbedding,
    ComplexDropout,
    ComplexBatchNorm1d,
    ComplexMult,
    ComplexAdd,
    ComplexSubstract,
    ComplexDiv,
    ComplexConjugate
)


class BooleanEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim, num_channels, norm_num_channels=2):
        super(BooleanEmbedding, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.embeddings = nn.ModuleList([ComplexEmbedding(num_entities, embedding_dim, norm_num_channels) for _ in range(num_channels)])

    def forward(self, idx):
        embedings = []
        for embedding in self.embeddings:
            embedings.append(embedding(idx))
        return tuple(embedings)

    def init(self):
        for embedding in self.embeddings:
            embedding.init()

    def get_embeddings(self):
        return [embedding.get_embeddings() for embedding in self.embeddings]

    def get_cat_embedding(self):
        return torch.cat([embedding.get_cat_embedding() for embedding in self.embeddings], 1)


class BooleanDropout(nn.Module):
    def __init__(self, dropout_rate_list: List[List[float]]):
        super(BooleanDropout, self).__init__()
        self.dropout_rate_list = dropout_rate_list
        self.dropouts = nn.ModuleList([ComplexDropout(dropout_rate) for dropout_rate in dropout_rate_list])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.dropouts[idx](complex_number))
        return tuple(out)


class BooleanBatchNorm1d(nn.Module):
    def __init__(self, embedding_dim, num_channels, norm_num_channels=2):
        super(BooleanBatchNorm1d, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.batch_norms = nn.ModuleList([ComplexBatchNorm1d(embedding_dim, norm_num_channels) for _ in range(num_channels)])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.batch_norms[idx](complex_number))
        return tuple(out)


class BooleanNorm(nn.Module):
    def forward(self,
                e: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
                )-> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        (alpha_a, alpha_b), (beta_a, beta_b) = e
        length = torch.sqrt(alpha_a ** 2 + alpha_b ** 2 + beta_a ** 2 + beta_b ** 2)
        return (alpha_a / length, alpha_b / length), (beta_a / length, beta_b / length)


class BooleanMult(nn.Module):
    """
    U[r] = [[r_a, -^r_b],
            [r_b, ^r_a ]]  ^ is conj
    h = [h_a, h_b]

    h_a, h_b in CP^d
    r_a, r_b in CP^d

    h * r = U[r] * h = [r_a * h_a + -^r_b * h_b, r_b * h_a + ^r_a * h_b]
    """

    def __init__(self, norm_flag=False):
        super(BooleanMult, self).__init__()
        self.norm_flag = norm_flag
        self.complex_mul = ComplexMult(False)
        self.complex_add = ComplexAdd()
        self.complex_sub = ComplexSubstract()
        self.complex_div = ComplexDiv()
        self.complex_conj = ComplexConjugate()
        self.bool_norm = BooleanNorm()

    def forward(self,
                h: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                r: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        h_a, h_b = self.bool_norm(h)
        r_a, r_b = self.bool_norm(r)
        a = self.complex_sub(self.complex_mul(h_a, r_a), self.complex_mul(h_b, self.complex_conj(r_b)))
        b = self.complex_add(self.complex_mul(h_a, r_b), self.complex_mul(h_b, self.complex_conj(r_a)))
        return a, b


class BooleanScoringAll(nn.Module):
    def forward(self, complex_numbers, embeddings):
        e_a, e_b = embeddings
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            t_a, t_b = complex_number
            a = torch.mm(t_a, e_a[idx].transpose(1, 0))
            b = torch.mm(t_b, e_b[idx].transpose(1, 0))
            out.append((a, b))
        return tuple(out)
