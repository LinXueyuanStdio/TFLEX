import math

import torch
from torch import nn

from toolbox.nn.ComplexEmbedding import ComplexEmbedding


class BracketEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim):
        super().__init__()
        self.bra = nn.Embedding(num_entities, embedding_dim)
        self.ket = nn.Embedding(num_entities, embedding_dim)

    def forward(self, index):
        return self.bra(index), self.ket(index)

    def init(self):
        nn.init.xavier_normal_(self.bra.weight.data)
        nn.init.xavier_normal_(self.ket.weight.data)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe(pos, 2i) = sin( pos / 10000^(2i/d) )
        #             = sin( pos * exp(log(10000) * -2i/d))
        #             = sin( pos * exp( 2i * -log(10000)/d ))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


# class RotatePositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len=512):
#         super(RotatePositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         # pe(pos, 2i) = sin( pos / 10000^(2i/d) )
#         #             = sin( pos * exp(log(10000) * -2i/d))
#         #             = sin( pos * exp( 2i * -log(10000)/d ))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#       # https://kexue.fm/archives/8265
#         # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         sin, cos = sinusoidal_pos.chunk(2, dim=-1)
#         # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
#         # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
#         # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
#         rotate_half_query_layer = torch.stack(
#             [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
#         ).reshape_as(query_layer)
#         query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
#         # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
#         rotate_half_key_layer = torch.stack(
#             [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
#         ).reshape_as(key_layer)
#         key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
#         if value_layer is not None:
#             # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
#             rotate_half_value_layer = torch.stack(
#                 [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
#             ).reshape_as(value_layer)
#             value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
#             return query_layer, key_layer, value_layer
#         return query_layer, key_layer
#
#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return x


class ComplexScoringAll(nn.Module):
    def forward(self, complex_numbers, embeddings):
        """
        complex_numbers = [(Bxd)]
        embeddings = [(Nxd)]
        return [(BxN)]
        """
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            a = complex_number
            b = embeddings[idx]
            ans = torch.mm(a, b.transpose(1, 0))  # 内积
            # ans = torch.cdist(a, b, p=1.0, compute_mode="donot_use_mm_for_euclid_dist")
            out.append(ans)
        return tuple(out)


class BracketTransformer(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim,
                 input_dropout=0.1, hidden_dropout=0.3,
                 feedforward_dim=2048, nhead=4, num_layers=2,
                 use_position_embedding=False,
                 ):
        super(BracketTransformer, self).__init__()
        self.E = ComplexEmbedding(num_entities, embedding_dim)
        self.R = ComplexEmbedding(num_relations, embedding_dim)
        self.answer_entity = ComplexEmbedding(1, embedding_dim)
        self.op_equal = ComplexEmbedding(1, embedding_dim)
        self.use_position_embedding = use_position_embedding
        self.pos_encoder = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=input_dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * 2),
        )

        self.scoring_all = ComplexScoringAll()
        self.dropout = nn.Dropout(hidden_dropout)
        self.m = nn.PReLU()

        self.b_x = nn.Parameter(torch.zeros(num_entities))
        self.b_y = nn.Parameter(torch.zeros(num_entities))
        self.bce = nn.BCELoss()

    def init(self):
        self.E.init()
        self.R.init()
        self.answer_entity.init()
        self.op_equal.init()

    def forward(self, h, r):
        h = self.E(h)  # Bxd, Bxd
        r = self.R(r)  # Bxd, Bxd
        h_bra, h_ket = h
        r_bra, r_ket = r
        B = h_bra.size(0)
        idx = torch.LongTensor([0] * B).to(h_bra.device)
        answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
        # op_equal_bra, op_equal_ket = self.op_equal(idx)
        # embed_list = [r_bra, h_bra, h_ket, r_ket, op_equal_bra, op_equal_ket, answer_entity_bra, answer_entity_ket]
        # embed_list = [answer_entity_bra, r_bra, h_bra]
        embed_list = [answer_entity_bra, answer_entity_ket, r_bra, h_bra, h_ket, r_ket]
        l = [i.view(1, B, -1) for i in embed_list]
        # for i in l:
        #     print(i.shape)
        src = torch.cat(l, dim=0)
        if self.use_position_embedding:
            src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        # print(out.shape)

        t_bra, t_ket = torch.chunk(out[:2, :, :], 2, dim=0)
        # print(t_bra.shape, t_ket.shape)
        t_bra = t_bra.view(B, -1)
        t_ket = t_ket.view(B, -1)
        t_bra, t_ket = torch.chunk(self.mlp(torch.cat([t_bra, t_ket], dim=-1)), 2, dim=-1)
        t = (t_bra, t_ket)
        y_a, y_b = self.scoring_all(t, self.E.get_embeddings())
        # print(y_a.shape, y_b.shape)
        y_a = y_a + self.b_x.expand_as(y_a)
        y_b = y_b + self.b_y.expand_as(y_b)

        y_a = torch.sigmoid(y_a)
        y_b = torch.sigmoid(y_b)

        return y_a, y_b

    def loss(self, predictions, target):
        y_a, y_b = predictions
        return self.bce(y_a, target) + self.bce(y_b, target)


if __name__ == "__main__":
    B = 5
    d = 32
    E = 10
    R = 10
    h = torch.randint(0, 5, size=(B,)).long()
    r = torch.randint(0, 5, size=(B,)).long()
    model = BracketTransformer(E, R, d)
    y = torch.randint(0, 2, size=(B, E)).float()
    target = model(h, r)
    print(model.loss(target, y))
