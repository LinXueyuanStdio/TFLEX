from typing import List

import torch
from torch import nn

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexBatchNorm1d


class MobiusEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim, num_channels, norm_num_channels=2):
        super(MobiusEmbedding, self).__init__()
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


class MobiusDropout(nn.Module):
    def __init__(self, dropout_rate_list: List[List[float]]):
        super(MobiusDropout, self).__init__()
        self.dropout_rate_list = dropout_rate_list
        self.dropouts = nn.ModuleList([ComplexDropout(dropout_rate) for dropout_rate in dropout_rate_list])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.dropouts[idx](complex_number))
        return tuple(out)


class MobiusBatchNorm1d(nn.Module):
    def __init__(self, embedding_dim, num_channels, norm_num_channels=2):
        super(MobiusBatchNorm1d, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.batch_norms = nn.ModuleList([ComplexBatchNorm1d(embedding_dim, norm_num_channels) for _ in range(num_channels)])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.batch_norms[idx](complex_number))
        return tuple(out)
