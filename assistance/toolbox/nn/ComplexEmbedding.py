from typing import List, Tuple

import torch
from torch import nn

ComplexNum = Tuple[torch.Tensor, torch.Tensor]


class ComplexEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim, num_channels=2):
        super(ComplexEmbedding, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.embeddings = nn.ModuleList([nn.Embedding(num_entities, embedding_dim) for _ in range(num_channels)])

    def forward(self, idx):
        embedings = []
        for embedding in self.embeddings:
            embedings.append(embedding(idx))
        return tuple(embedings)

    def init(self):
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight.data)

    def get_embeddings(self):
        return [embedding.weight for embedding in self.embeddings]

    def get_cat_embedding(self):
        return torch.cat([embedding.weight.data for embedding in self.embeddings], 1)

    def scoring_all(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(complex_numbers):
            ans = torch.mm(complex_number, self.embeddings[idx].weight.transpose(1, 0))
            out.append(ans)
        return tuple(out)


class ComplexScoringAll(nn.Module):
    def forward(self, complex_numbers, embeddings):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            ans = torch.mm(complex_number, embeddings[idx].transpose(1, 0))
            out.append(ans)
        return tuple(out)


class ComplexAlign(nn.Module):
    def dis(self, x, y):
        return torch.sum(x * y, dim=-1)

    def forward(self, complex_numbers, embeddings):
        """
        complex_numbers: (a,b,c,d,...) for z=a+bi+cj+dk+... each real number is B x d_e
        embeddings:      (a,b,c,d,...) for z=a+bi+cj+dk+... each real number is B x d_e
        """
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            ans = self.dis(complex_number, embeddings[idx])
            out.append(ans)
        return tuple(out)


class ComplexDropout(nn.Module):
    def __init__(self, dropout_rate_list: List[float]):
        super(ComplexDropout, self).__init__()
        self.dropout_rate_list = dropout_rate_list
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for dropout_rate in dropout_rate_list])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.dropouts[idx](complex_number))
        return tuple(out)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, embedding_dim, num_channels=2):
        super(ComplexBatchNorm1d, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(embedding_dim) for _ in range(num_channels)])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.batch_norms[idx](complex_number))
        return tuple(out)


class ComplexMult(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h * r = (h_a * r_a - h_b * r_b) + (h_a * r_b + h_b * r_a) i

    h in C^d
    r in C^d
    """

    def __init__(self, norm_flag=False):
        super(ComplexMult, self).__init__()
        self.flag_hamilton_mul_norm = norm_flag

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        if self.flag_hamilton_mul_norm:
            # Normalize the relation to eliminate the scaling effect
            r_norm = torch.sqrt(r_a ** 2 + r_b ** 2)
            r_a = r_a / r_norm
            r_b = r_b / r_norm
        t_a = h_a * r_a - h_b * r_b
        t_b = h_a * r_b + h_b * r_a
        return t_a, t_b


class ComplexAdd(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h + r = (h_a + r_a) + (h_b + r_b) i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexAdd, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        t_a = h_a + r_a
        t_b = h_b + r_b
        return t_a, t_b


class ComplexConjugate(nn.Module):
    """
    h = h_a + h_b i
    ^h = h_a - h_b i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexConjugate, self).__init__()

    def forward(self, h):
        h_a, h_b = h
        return h_a, -h_b


class ComplexSubstract(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h - r = (h_a - r_a) + (h_b - r_b) i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexSubstract, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        t_a = h_a - r_a
        t_b = h_b - r_b
        return t_a, t_b


class ComplexDiv(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h / r = [(h_a * r_a + h_b * r_b) / (r_a ^2 + r_b ^2)] + [(h_b * r_a - h_a * r_b) / (r_a ^2 + r_b ^2)] i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexDiv, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        r_norm = torch.sqrt(r_a ** 2 + r_b ** 2)

        t_a = (h_a * r_a + h_b * r_b) / r_norm
        t_b = (h_b * r_a - h_a * r_b) / r_norm
        return t_a, t_b
