"""
@date: 2021/10/14
@description: null
"""
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from BooleanEmbedding import BooleanBatchNorm1d, BooleanDropout, BooleanEmbedding, BooleanScoringAll, BooleanNorm, BooleanMult
from toolbox.nn.ComplexEmbedding import ComplexAlign, ComplexMult
from toolbox.nn.Regularizer import N3

ComplexNum = Tuple[torch.Tensor, torch.Tensor]
QubitLevel1 = Tuple[ComplexNum, ComplexNum]
QubitLevel2 = Tuple[ComplexNum, ComplexNum, ComplexNum, ComplexNum]
QubitLevel3 = Tuple[ComplexNum, ComplexNum, ComplexNum, ComplexNum,
                    ComplexNum, ComplexNum, ComplexNum, ComplexNum]
QubitLevel4 = Tuple[ComplexNum, ComplexNum, ComplexNum, ComplexNum,
                    ComplexNum, ComplexNum, ComplexNum, ComplexNum,
                    ComplexNum, ComplexNum, ComplexNum, ComplexNum,
                    ComplexNum, ComplexNum, ComplexNum, ComplexNum]
Qubit = QubitLevel1


class CNOT(nn.Module):

    def __init__(self, device="cuda:0"):
        super(CNOT, self).__init__()
        self.CNOT = torch.FloatTensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]).to(device)
        self.mul = ComplexMult(False)

    def forward(self, x: Qubit, y: Qubit) -> QubitLevel2:
        x_a, x_b = x
        y_a, y_b = y
        out_a: ComplexNum = self.mul(x_a, y_a)
        out_b: ComplexNum = self.mul(x_a, y_b)
        out_c: ComplexNum = self.mul(x_b, y_a)
        out_d: ComplexNum = self.mul(x_b, y_b)
        return out_a, out_b, out_d, out_c


class CCNOT(nn.Module):

    def __init__(self, device="cuda:0"):
        super(CCNOT, self).__init__()
        self.CCNOT = torch.FloatTensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]).to(device)
        self.mul = ComplexMult(False)

    def forward(self, x: Qubit, y: Qubit, z: Qubit) -> QubitLevel3:
        x_a, x_b = x
        y_a, y_b = y
        z_a, z_b = z
        tmp_a = self.mul(x_a, y_a)
        tmp_b = self.mul(x_a, y_b)
        tmp_c = self.mul(x_b, y_a)
        tmp_d = self.mul(x_b, y_b)
        out_a = self.mul(tmp_a, z_a)
        out_b = self.mul(tmp_b, z_a)
        out_c = self.mul(tmp_c, z_a)
        out_d = self.mul(tmp_d, z_a)
        out_e = self.mul(tmp_a, z_b)
        out_f = self.mul(tmp_b, z_b)
        out_g = self.mul(tmp_c, z_b)
        out_h = self.mul(tmp_d, z_b)
        return out_a, out_b, out_c, out_d, out_e, out_f, out_h, out_g


class NOT(nn.Module):

    def __init__(self, device="cuda:0"):
        super(NOT, self).__init__()
        # self.NOT = torch.FloatTensor([
        #     [0, 1],
        #     [1, 0],
        # ]).to(device)

    def forward(self, x: Qubit) -> QubitLevel1:
        x_a, x_b = x
        return x_b, x_a


class Identity(nn.Module):

    def forward(self, x: Qubit) -> QubitLevel1:
        return x


class Y(nn.Module):

    def __init__(self, device="cuda:0"):
        super(Y, self).__init__()
        # self.Y = torch.FloatTensor([
        #     [0, -1],
        #     [1, 0],
        # ]).to(device)

    def forward(self, x: Qubit) -> QubitLevel1:
        x_a, x_b = x
        x_b_a, x_b_b = x_b
        neg_x_b = (-x_b_a, -x_b_b)
        return neg_x_b, x_a


class Z(nn.Module):

    def __init__(self, device="cuda:0"):
        super(Z, self).__init__()
        # self.Z = torch.FloatTensor([
        #     [1, 0],
        #     [0, -1],
        # ]).to(device)

    def forward(self, x: Qubit) -> QubitLevel1:
        x_a, x_b = x
        x_b_a, x_b_b = x_b
        neg_x_b = (-x_b_a, -x_b_b)
        return x_a, neg_x_b


class Logic1p(nn.Module):

    def __init__(self, norm_flag=False):
        super(Logic1p, self).__init__()
        self.mul = BooleanMult(norm_flag)

    def forward(self, h: Qubit, r) -> QubitLevel1:
        t = self.mul(h, r)
        return t


class Logic2i(nn.Module):
    """(x and y) = Logic2i(x, y)"""

    def __init__(self, edim, device="cuda:0"):
        super(Logic2i, self).__init__()
        # |0> = 1 |0> + 0 |1>
        self.one: torch.Tensor = torch.ones((edim, 1)).to(device)
        self.zero: torch.Tensor = torch.zeros((edim, 1)).to(device)
        self.complexOne: ComplexNum = (self.one, self.zero)
        self.complexZero: ComplexNum = (self.zero, self.zero)
        self.zeroQubit: Qubit = (self.complexOne, self.complexZero)
        self.mul = ComplexMult(False)

    def forward(self, x: Qubit, y: Qubit) -> QubitLevel3:
        # CCNOT(|x, y, 0>) = |x, y, (x and y) >
        # |x> --[C]-- |x>
        # |y> --[C]-- |y>
        # |0> --[X]-- |(x and y)>
        x_a, x_b = x
        y_a, y_b = y
        tmp_a = self.mul(x_a, y_a)
        tmp_b = self.mul(x_a, y_b)
        tmp_c = self.mul(x_b, y_a)
        tmp_d = self.mul(x_b, y_b)
        out_a: ComplexNum = tmp_a
        out_b: ComplexNum = tmp_b
        out_c: ComplexNum = tmp_c
        out_d: ComplexNum = self.complexZero
        out_e: ComplexNum = self.complexZero
        out_f: ComplexNum = self.complexZero
        out_g: ComplexNum = self.complexZero
        out_h: ComplexNum = tmp_d
        return out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h


class Logic2u(nn.Module):
    """(x or y) = Logic2u(x, y)"""

    def __init__(self, edim, device="cuda:0"):
        super(Logic2u, self).__init__()
        # |0> = 1 |0> + 0 |1>
        self.one: torch.Tensor = torch.ones((edim, 1)).to(device)
        self.zero: torch.Tensor = torch.zeros((edim, 1)).to(device)
        self.complexOne: ComplexNum = (self.one, self.zero)
        self.complexZero: ComplexNum = (self.zero, self.zero)
        self.zeroQubit: Qubit = (self.complexOne, self.complexZero)
        self.mul = ComplexMult(False)
        self.X = NOT(device)

    def forward(self, x: Qubit, y: Qubit) -> QubitLevel3:
        # |x> --[X]--[C]--[X]-- |x>
        # |y> --[X]--[C]--[X]-- |y>
        # |0> -------[X]--[X]-- |(x or y)>
        # TODO 没实现完，等
        x_a, x_b = self.X(x)
        y_a, y_b = self.X(y)
        tmp_a = self.mul(x_a, y_a)
        tmp_b = self.mul(x_a, y_b)
        tmp_c = self.mul(x_b, y_a)
        tmp_d = self.mul(x_b, y_b)
        out_a: ComplexNum = tmp_a
        out_b: ComplexNum = tmp_b
        out_c: ComplexNum = tmp_c
        out_d: ComplexNum = self.complexZero
        out_e: ComplexNum = self.complexZero
        out_f: ComplexNum = self.complexZero
        out_g: ComplexNum = self.complexZero
        out_h: ComplexNum = tmp_d
        return out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h


class LogicXor(nn.Module):
    """(x xor y) = LogicXor(x, y)"""

    def __init__(self, edim, device="cuda:0"):
        super(LogicXor, self).__init__()
        # |0> = 1 |0> + 0 |1>
        self.one: torch.Tensor = torch.ones((edim, 1)).to(device)
        self.zero: torch.Tensor = torch.zeros((edim, 1)).to(device)
        self.complexOne: ComplexNum = (self.one, self.zero)
        self.complexZero: ComplexNum = (self.zero, self.zero)
        self.zeroQubit: Qubit = (self.complexOne, self.complexZero)
        self.oneQubit: Qubit = (self.complexZero, self.complexOne)
        self.mul = ComplexMult(False)
        self.X = NOT(device)

    def forward(self, x: Qubit, y: Qubit) -> QubitLevel3:
        # |1> --[C]-- |1>
        # |x> --[C]-- |x>
        # |y> --[X]-- |(x xor y)>
        x_a, x_b = x
        y_a, y_b = y
        tmp_a = self.mul(x_a, y_a)
        tmp_b = self.mul(x_a, y_b)
        tmp_c = self.mul(x_b, y_a)
        tmp_d = self.mul(x_b, y_b)
        out_a: ComplexNum = self.complexZero
        out_b: ComplexNum = self.complexZero
        out_c: ComplexNum = tmp_a
        out_d: ComplexNum = tmp_c
        out_e: ComplexNum = self.complexZero
        out_f: ComplexNum = self.complexZero
        out_g: ComplexNum = tmp_d
        out_h: ComplexNum = tmp_b
        return out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h


class Logic3i(nn.Module):
    """(x and y and z) = Logic3i(x, y, z)"""

    def __init__(self, device="cuda:0"):
        super(Logic3i, self).__init__()
        self.zero = torch

    def forward(self, x: Qubit, y: Qubit, z: Qubit) -> QubitLevel4:
        pass


class BoolE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3, regularization_weight=0.1):
        super(BoolE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.bce = nn.BCELoss()
        self.E = BooleanEmbedding(self.num_entities, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.R = BooleanEmbedding(self.num_relations, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.E_dropout = BooleanDropout([[input_dropout, input_dropout]] * 2)
        self.R_dropout = BooleanDropout([[input_dropout, input_dropout]] * 2)
        self.hidden_dp = BooleanDropout([[hidden_dropout, hidden_dropout]] * 2)
        self.E_bn = BooleanBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = BooleanBatchNorm1d(self.embedding_dim, 2)
        self.b_x = nn.Parameter(torch.zeros(num_entities))
        self.b_y = nn.Parameter(torch.zeros(num_entities))
        self.norm = BooleanNorm()

        self.mul = BooleanMult(norm_flag)
        self.scoring_all = BooleanScoringAll()
        self.align = ComplexAlign()
        self.regularizer = N3(regularization_weight)

    def forward(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        return self.forward_head_batch(h_idx, r_idx)

    def forward_head_batch(self, e1_idx, rel_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x) | x in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(e1_idx)
        r = self.R(rel_idx)
        h = self.norm(h)
        t = self.mul(h, r)
        t = self.norm(t)

        score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(self.E_bn(self.norm(self.E.get_embeddings()))))
        score_a_a, score_a_b = score_a
        y_a = score_a_a + score_a_b
        y_a = y_a + self.b_x.expand_as(y_a)

        score_b_a, score_b_b = score_b
        y_b = score_b_a + score_b_b
        y_b = y_b + self.b_y.expand_as(y_b)

        y_a = torch.sigmoid(y_a)
        y_b = torch.sigmoid(y_b)

        return y_a, y_b

    def loss(self, target, y):
        y_a, y_b = target
        return self.bce(y_a, y) + self.bce(y_b, y)

    def regular_loss(self, e1_idx, rel_idx):
        h = self.E(e1_idx)
        r = self.R(rel_idx)
        (h_a, h_a_i), (h_b, h_b_i) = h
        (r_a, r_a_i), (r_b, r_b_i) = r
        factors = (
            torch.sqrt(h_a ** 2 + h_a_i ** 2 + h_b ** 2 + h_b_i ** 2),
            torch.sqrt(r_a ** 2 + r_a_i ** 2 + r_b ** 2 + r_b_i ** 2),
        )
        regular_loss = self.regularizer(factors)
        return regular_loss

    def reverse_loss(self, e1_idx, rel_idx, max_relation_idx):
        h = self.E(e1_idx)
        h_a, h_b = h
        h = (h_a.detach(), h_b.detach())

        r = self.R(rel_idx)
        reverse_rel_idx = (rel_idx + max_relation_idx) % (2 * max_relation_idx)

        t = self.mul(h, r)
        reverse_r = self.R(reverse_rel_idx)
        reverse_t = self.mul(t, reverse_r)
        reverse_a, reverse_b = self.align(reverse_t, h)  # a + b i
        reverse_score = reverse_a + reverse_b
        reverse_score = torch.mean(F.relu(reverse_score))

        return reverse_score

    def query_loss(self, positive_sample, negative_sample, subsampling_weight,
                   batch_queries_dict: Dict[Tuple[Union[Tuple[str], str]], torch.Tensor],
                   batch_idxs_dict: Dict[Tuple[Union[Tuple[str], str]], List[List[int]]]):
        # 1. 用 batch_queries_dict 嵌入查询为 量子叠加态（编码后的状态）
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []  # 处理 一般的查询
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []  # 处理 并查询
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query(self.transform_union_query(batch_queries_dict[query_structure],
                                                                query_structure),
                                     self.transform_union_structure(query_structure),
                                     0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query(batch_queries_dict[query_structure],
                                                                      query_structure,
                                                                      0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)  # (B, 1, d)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)  # (B, 1, d)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)  # Beta(B, 1, d)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0] // 2, 2, 1, -1)  # (B/2, 2, 1, d)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0] // 2, 2, 1, -1)  # (B/2, 2, 1, d)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)  # Beta(B/2, 2, 1, d)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_sample) != type(None):
            # 2.1 计算 一般的查询
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.query_logit(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            # 2.2 计算 并查询
            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.query_logit(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        # 3. 计算负例损失
        if type(negative_sample) != type(None):
            # 3.1 计算 一般的查询
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.query_logit(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            # 3.2 计算 并查询
            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.query_logit(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def embed_query(self, queries: torch.Tensor, query_structure: Tuple[Union[Tuple[str], str]], idx: int):
        """
        迭代嵌入
        例子：(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
        B = 2, queries=[[1]]
        Iterative embed a batch of queries with same structure using BetaE
        queries:(B, L): a flattened batch of queries (all queries are of query_structure), B is batch size, L is length of queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:
            # whether the current query tree has merged to one branch and only need to do relation traversal,
            # e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            # 这一类如下
            #     ('e', ('r',)): '1p',
            #     ('e', ('r', 'r')): '2p',
            #     ('e', ('r', 'r', 'r')): '3p',
            #     ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
            #     ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
            # 都是左边是[实体, 中间推理状态]，右边是[关系, 否运算]，只用把状态沿着运算的方向前进一步
            # 所以对 query_structure 的索引只有 0 (左) 和 -1 (右)
            if query_structure[0] == 'e':
                # 嵌入实体
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                # 嵌入中间推理状态
                alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):  # 有若干个关系或否运算
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1. / embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            # 这一类比较复杂，表示最后一个运算是且运算
            #     (('e', ('r',)), ('e', ('r',))): '2i',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
            #     (('e', ('r', 'r')), ('e', ('r',))): 'pi',
            #     (('e', ('r',)), ('e', ('r', 'n'))): '2in',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
            #     (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
            #     (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):  # 把内部每个子结构都嵌入了，再执行 且运算
                alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def query_logit(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def transform_union_query(self, queries, query_structure: Tuple[Union[Tuple[str], str]]) -> torch.Tensor:
        """
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        """
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure: Tuple[Union[Tuple[str], str]]) -> Tuple[Union[Tuple[str], str]]:
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def init(self):
        self.E.init()
        self.R.init()


if __name__ == "__main__":
    h = torch.LongTensor([[i] for i in range(5)])
    r = torch.LongTensor([[i] for i in range(5)])
    model = BoolE(10, 10, 5)
    pred = model(h, r)
    print(pred)
    print(pred.shape)
    y = torch.rand_like(pred)
    print(model.loss(pred, y))
