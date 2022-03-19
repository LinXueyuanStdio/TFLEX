import math
import random
from typing import *

import torch
from torch import nn

from ComplexQueryData import QueryStructure
from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexScoringAll
import random

class BracketEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bra = nn.Embedding(num_entities, embedding_dim // 2)
        self.ket = nn.Embedding(num_entities, embedding_dim // 2)

    def forward(self, index):
        bra, ket = self.bra(index), self.ket(index)
        zeros = torch.zeros_like(bra, device=bra.device)
        bra = torch.cat([bra, zeros], dim=-1)
        ket = torch.cat([zeros, ket], dim=-1)
        return bra, ket

    def init(self):
        nn.init.xavier_normal_(self.bra.weight.data)
        nn.init.xavier_normal_(self.ket.weight.data)


class Bracket(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim,
                 gamma,
                 input_dropout=0.1, hidden_dropout=0.3,
                 feedforward_dim=2048, nhead=4, num_layers=2,
                 use_position_embedding=False,
                 test_batch_size=1, use_cuda=False, query_name_dict: Dict[QueryStructure, str] = None, center_reg=None, drop=0.):
        super(Bracket, self).__init__()
        self.E = ComplexEmbedding(nentity, hidden_dim)
        self.R = ComplexEmbedding(nrelation, hidden_dim)
        self.answer_entity = ComplexEmbedding(1, hidden_dim)
        self.negation = ComplexEmbedding(1, hidden_dim)
        self.intersection = ComplexEmbedding(1, hidden_dim)
        self.union = ComplexEmbedding(1, hidden_dim)

        self.use_position_embedding = use_position_embedding

        max_len = 64
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=input_dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scoring_all = ComplexScoringAll()
        self.dropout = nn.Dropout(hidden_dropout)
        self.m = nn.PReLU()

        self.b_x = nn.Parameter(torch.zeros(nentity))
        self.b_y = nn.Parameter(torch.zeros(nentity))
        self.bce = nn.BCELoss()
        # loss

        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.query_name_dict: Dict[QueryStructure, str] = query_name_dict

        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        if self.use_cuda:
            self.batch_entity_range = self.batch_entity_range.cuda()
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

    def init(self):
        self.E.init()
        self.R.init()
        self.answer_entity.init()
        self.negation.init()
        self.intersection.init()
        self.union.init()

    # implement formatting forward method
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_bracket(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_bracket(self, positive_sample, negative_sample, subsampling_weight,
                        batch_queries_dict: Dict[QueryStructure, torch.Tensor],
                        batch_idxs_dict: Dict[QueryStructure, List[List[int]]]):
        # 1. 用 batch_queries_dict 将 查询 嵌入
        all_idxs, all_bra, all_ket = [], [], []
        all_union_idxs, all_union_bra, all_union_ket = [], [], []
        device = self.b_x.device
        for query_structure in batch_queries_dict:
            # 用字典重新组织了嵌入，一个批次(BxL)只对应一种结构
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                t, _ = self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                             self.transform_union_structure(query_structure),
                                             0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                t_bra, t_ket = t
                all_union_bra.append(t_bra)
                all_union_ket.append(t_ket)
            else:
                t, _ = self.embed_query_cone(batch_queries_dict[query_structure],
                                             query_structure,
                                             0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                t_bra, t_ket = t
                all_bra.append(t_bra)
                all_ket.append(t_ket)

        if len(all_bra) > 0:
            all_bra = torch.cat(all_bra, dim=0).unsqueeze(1)  # (B, 1, d)
            all_ket = torch.cat(all_ket, dim=0).unsqueeze(1)  # (B, 1, d)
        if len(all_union_bra) > 0:
            all_union_bra = torch.cat(all_union_bra, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_ket = torch.cat(all_union_ket, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_bra = all_union_bra.view(all_union_bra.shape[0] // 2, 2, 1, -1)
            all_union_ket = all_union_ket.view(all_union_ket.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_sample) != type(None):
            # 2.1 计算 一般的查询
            if len(all_bra) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                y_bra, y_ket = self.E(positive_sample_regular)
                y_bra = y_bra.unsqueeze(1)
                y_ket = y_ket.unsqueeze(1)
                positive_logit = self.cal_logit_cone(y_bra, y_ket, all_bra, all_ket)
            else:
                positive_logit = torch.Tensor([]).to(device)

            # 2.1 计算 并查询
            if len(all_union_bra) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                y_bra, y_ket = self.E(positive_sample_union)
                y_bra = y_bra.unsqueeze(1).unsqueeze(1)
                y_ket = y_ket.unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_cone(y_bra, y_ket, all_union_bra, all_union_ket)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        # 3. 计算负例损失
        if type(negative_sample) != type(None):
            # 3.1 计算 一般的查询
            if len(all_bra) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                y_bra, y_ket = self.E(negative_sample_regular.view(-1))
                y_bra = y_bra.view(batch_size, negative_size, -1)
                y_ket = y_ket.view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_cone(y_bra, y_ket, all_bra, all_ket)
            else:
                negative_logit = torch.Tensor([]).to(device)

            # 3.1 计算 并查询
            if len(all_union_bra) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                y_bra, y_ket = self.E(negative_sample_union.view(-1))
                y_bra = y_bra.view(batch_size, 1, negative_size, -1)
                y_ket = y_ket.view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_cone(y_bra, y_ket, all_union_bra, all_union_ket)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def embed_query_cone(self, queries: torch.Tensor, query_structure, idx: int):
        """
        迭代嵌入
        例子：(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
        B = 2, queries=[[1]]
        Iterative embed a batch of queries with same structure using BetaE
        queries:(B, L): a flattened batch of queries (all queries are of query_structure), B is batch size, L is length of queries
        """
        query_name = self.query_name_dict[query_structure]
        encode_list = self.encode(query_name, queries)
        if len(encode_list) > 0:
            B = queries.size(0)
            l = [i.view(1, B, -1) for i in encode_list]
            src = torch.cat(l, dim=0)
            if self.use_position_embedding:
                src = src + self.pe.to(src.device)[:src.size(0)]
            out = self.transformer_encoder(src)
            t_bra, t_ket = torch.chunk(out[:2, :, :], 2, dim=0)
            t = (t_bra.view(B, -1), t_ket.view(B, -1))
        else:
            t = None
        return t, idx

    def encode(self, query_name: str, queries: torch.Tensor):
        """
        '1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM'
        """
        if query_name == "1p":  # ('e', ('r',))
            h1_idx, r1_idx = queries[:, 0], queries[:, 1]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            encode_list = [
                answer_entity_bra, answer_entity_ket,
                r1_bra, h1_bra, h1_ket, r1_ket,
            ]
        elif query_name == "2p":  # ('e', ('r', 'r'))
            h1_idx, r1_idx, r2_idx = queries[:, 0], queries[:, 1], queries[:, 2]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            encode_list = [
                answer_entity_bra, answer_entity_ket,
                r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
            ]
        elif query_name == "3p":  # ('e', ('r', 'r', 'r'))
            h1_idx, r1_idx, r2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            # [ ] [  [  [  [  ]  ]  ]  ]
            # a a r3 r2 r1 h1 h1 r1 r2 r3
            encode_list = [
                answer_entity_bra, answer_entity_ket,
                r3_bra, r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket, r3_ket,
            ]
        elif query_name == "2i":  # (('e', ('r',)), ('e', ('r',)))
            h1_idx, r1_idx, h2_idx, r2_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
        elif query_name == "3i":  # (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))
            h1_idx, r1_idx, h2_idx, r2_idx, h3_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 4], queries[:, 5]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            h3_bra, h3_ket = self.E(h3_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            key = random.randint(0, 6)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    inter_ket,
                ]
            elif key == 1:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    inter_ket,
                ]
            elif key == 1:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                ]
            elif key == 2:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
            elif key == 3:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    inter_ket,
                ]
            elif key == 4:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                ]
        elif query_name == "ip":  # ((('e', ('r',)), ('e', ('r',))), ('r',))
            h1_idx, r1_idx, h2_idx, r2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 4]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                    r3_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                    r3_ket,
                ]
        elif query_name == "pi":  # (('e', ('r', 'r')), ('e', ('r',)))
            h1_idx, r1_idx, r2_idx, h2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 4]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    inter_ket,
                ]
        elif query_name == "2in":  # (('e', ('r',)), ('e', ('r', 'n')))
            h1_idx, r1_idx, h2_idx, r2_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
        elif query_name == "3in":  # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
            h1_idx, r1_idx, h2_idx, r2_idx, h3_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 4], queries[:, 5]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            h3_bra, h3_ket = self.E(h3_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 6)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    inter_ket,
                ]
            elif key == 1:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    inter_ket,
                ]
            elif key == 1:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                ]
            elif key == 2:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
            elif key == 3:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    inter_ket,
                ]
            elif key == 4:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    neg_bra,
                    r3_bra, h3_bra, h3_ket, r3_ket,
                    neg_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    inter_ket,
                ]
        elif query_name == "inp":  # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
            h1_idx, r1_idx, h2_idx, r2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 5]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    inter_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    inter_ket,
                    r3_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    inter_bra,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    inter_ket,
                    r3_ket,
                ]
        elif query_name == "pin":  # (('e', ('r', 'r')), ('e', ('r', 'n')))
            h1_idx, r1_idx, r2_idx, h2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 4]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    neg_bra,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    neg_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    neg_bra,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    neg_ket,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    inter_ket,
                ]
        elif query_name == "pni":  # (('e', ('r', 'r', 'n')), ('e', ('r',)))
            h1_idx, r1_idx, r2_idx, h2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 4], queries[:, 5]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    neg_bra,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    neg_ket,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    inter_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    inter_bra,
                    r3_bra, h2_bra, h2_ket, r3_ket,
                    neg_bra,
                    r2_bra, r1_bra, h1_bra, h1_ket, r1_ket, r2_ket,
                    neg_ket,
                    inter_ket,
                ]
        elif query_name == "2u-DNF":  # (('e', ('r',)), ('e', ('r',)), ('u',))
            h1_idx, r1_idx, h2_idx, r2_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            union_bra, union_ket = self.union(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    union_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    union_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    union_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    union_ket,
                ]
        elif query_name == "up-DNF":  # ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
            h1_idx, r1_idx, h2_idx, r2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 2], queries[:, 3], queries[:, 5]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            union_bra, union_ket = self.union(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    union_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    union_ket,
                    r3_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    union_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    union_ket,
                    r3_ket,
                ]
        elif query_name == "2u-DM":  # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',))
            h1_idx, r1_idx, h2_idx, r2_idx = queries[:, 0], queries[:, 1], queries[:, 3], queries[:, 4]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    neg_bra,
                    inter_bra,
                    neg_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_ket,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    inter_ket,
                    neg_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    neg_bra,
                    inter_bra,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    neg_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_ket,
                    inter_ket,
                    neg_ket,
                ]
        elif query_name == "up-DM":  # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r'))
            h1_idx, r1_idx, h2_idx, r2_idx, r3_idx = queries[:, 0], queries[:, 1], queries[:, 3], queries[:, 4], queries[:, 7]
            h1_bra, h1_ket = self.E(h1_idx)  # Bxd, Bxd
            r1_bra, r1_ket = self.R(r1_idx)  # Bxd, Bxd
            h2_bra, h2_ket = self.E(h2_idx)  # Bxd, Bxd
            r2_bra, r2_ket = self.R(r2_idx)  # Bxd, Bxd
            r3_bra, r3_ket = self.R(r3_idx)  # Bxd, Bxd
            B = h1_bra.size(0)
            idx = torch.LongTensor([0] * B).to(h1_bra.device)
            answer_entity_bra, answer_entity_ket = self.answer_entity(idx)
            inter_bra, inter_ket = self.intersection(idx)
            neg_bra, neg_ket = self.negation(idx)
            key = random.randint(0, 2)
            if key == 0:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    neg_bra,
                    inter_bra,
                    neg_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_ket,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    inter_ket,
                    neg_ket,
                    r3_ket,
                ]
            else:
                encode_list = [
                    answer_entity_bra, answer_entity_ket,
                    r3_bra,
                    neg_bra,
                    inter_bra,
                    neg_bra,
                    r2_bra, h2_bra, h2_ket, r2_ket,
                    neg_ket,
                    neg_bra,
                    r1_bra, h1_bra, h1_ket, r1_ket,
                    neg_ket,
                    inter_ket,
                    neg_ket,
                    r3_ket,
                ]
        else:
            encode_list = []
        return encode_list

    # implement distance function
    def distance(self, entity_bracket, query_bracket):
        distance = torch.norm(entity_bracket - query_bracket, p=1, dim=-1)
        return distance

    def cal_logit_cone(self, y_bra, y_ket, query_bra, query_ket):
        distance_1 = self.distance(y_bra, query_bra)
        distance_2 = self.distance(y_ket, query_ket)
        logit = self.gamma - distance_1 * self.modulus - distance_2 * self.modulus
        return logit

    def transform_union_query(self, queries, query_structure: QueryStructure):
        """
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        """
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure: QueryStructure) -> QueryStructure:
        if self.query_name_dict[query_structure] == '2u-DNF':
            return 'e', ('r',)
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return 'e', ('r', 'r')


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
