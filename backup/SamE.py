"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/12/7
@description: axis和arg一样的取值范围
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ComplexQueryData import QueryStructure, query_name_dict

pi = 3.14159265358979323846


def convert_to_arg(x):
    # [0, 2pi]
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


# def convert_to_axis(x):
#     # [-pi, pi]
#     y = torch.tanh(x) * pi
#     return y
def convert_to_axis(x):
    return convert_to_arg(x)


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class Projection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(Projection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, q_axis, q_arg, r_axis, r_arg):
        x = torch.cat([q_axis + r_axis, q_arg + r_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis = convert_to_axis(axis)
        arg = convert_to_arg(arg)
        return axis, arg


class Intersection(nn.Module):
    def __init__(self, dim, drop):
        super(Intersection, self).__init__()
        self.dim = dim
        self.axis_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.axis_layer_2 = nn.Linear(self.dim, self.dim)
        self.arg_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.arg_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.axis_layer_1.weight)
        nn.init.xavier_uniform_(self.axis_layer_2.weight)
        nn.init.xavier_uniform_(self.arg_layer_1.weight)
        nn.init.xavier_uniform_(self.arg_layer_2.weight)

        self.drop = nn.Dropout(p=drop)

    def calculate_axis(self, axis, axis_attention):
        # 1.2 merged to (x, y)
        x_1 = torch.cos(axis)
        y_1 = torch.sin(axis)
        x_1 = torch.sum(axis_attention * x_1, dim=0)  # B x d
        y_1 = torch.sum(axis_attention * y_1, dim=0)  # B x d

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_1[torch.abs(x_1) < 1e-3] = 1e-3

        # 1.3 transformed back to axis
        axis = torch.atan(y_1 / x_1)

        # 1.4 normalize
        neg_x = x_1 < 0
        neg_y = y_1 < 0

        axis[neg_x & torch.logical_not(neg_y)] += pi
        axis[neg_x & neg_y] += pi
        axis[torch.logical_not(neg_x) & neg_y] += 2 * pi
        # [0, 2pi]
        return axis

    def forward(self, axis, arg):
        # axis: N x B x d
        # arg:  N x B x d
        logits = torch.cat([axis - arg, axis + arg], dim=-1)  # N x B x 2d

        # 1. calculate axis
        # 1.1 attention
        axis_attention = F.softmax(self.axis_layer_2(F.relu(self.axis_layer_1(logits))), dim=0)
        axis = self.calculate_axis(axis, axis_attention)

        # ==============================================
        # 2. calculate arg
        # DeepSets
        a = torch.mean(F.relu(self.arg_layer_1(logits)), dim=0)
        gate = torch.sigmoid(self.arg_layer_2(a))

        arg = self.drop(arg)
        arg, _ = torch.min(arg, dim=0)
        arg = arg * gate
        # arg_attention = F.softmax(self.arg_layer_2(F.relu(self.arg_layer_1(logits))), dim=0)
        # arg = self.calculate_axis(arg, arg_attention)

        return axis, arg


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_axis(self, axis):
        indicator_positive = axis >= 0
        indicator_negative = axis < 0
        axis[indicator_positive] = axis[indicator_positive] - pi
        axis[indicator_negative] = axis[indicator_negative] + pi
        return axis

    def forward(self, axis, arg):
        # axis = self.neg_axis(axis)
        indicator_positive = axis >= pi
        indicator_negative = axis < pi
        axis[indicator_positive] = axis[indicator_positive] - pi
        axis[indicator_negative] = axis[indicator_negative] + pi

        arg = pi - arg
        return axis, arg


class SamE(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 test_batch_size=1,
                 use_cuda=False,
                 query_name_dict=None,
                 center_reg=None, drop: float = 0.):
        super(SamE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        if self.use_cuda:
            self.batch_entity_range = self.batch_entity_range.cuda()

        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.cen = center_reg

        # entity only have axis but no arg
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True)  # axis 1 for entities
        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]
        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.cone_proj = Projection(self.entity_dim, 1600, 2)
        self.cone_intersection = Intersection(self.entity_dim, drop)
        self.cone_negation = Negation()

    def init(self):
        pass

    # implement formatting forward method
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_cone(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_cone(self, positive_sample, negative_sample, subsampling_weight,
                     batch_queries_dict: Dict[QueryStructure, torch.Tensor],
                     batch_idxs_dict: Dict[QueryStructure, List[List[int]]]):
        # 1. 用 batch_queries_dict 将 查询 嵌入
        all_idxs, all_axis, all_arg = [], [], []
        all_union_idxs, all_union_axis, all_union_arg = [], [], []
        for query_structure in batch_queries_dict:
            # 用字典重新组织了嵌入，一个批次(BxL)只对应一种结构
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis, arg, _ = self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                                     self.transform_union_structure(query_structure),
                                                     0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis.append(axis)
                all_union_arg.append(arg)
            else:
                axis, arg, _ = self.embed_query_cone(batch_queries_dict[query_structure],
                                                     query_structure,
                                                     0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis.append(axis)
                all_arg.append(arg)

        if len(all_axis) > 0:
            all_axis = torch.cat(all_axis, dim=0).unsqueeze(1)  # (B, 1, d)
            all_arg = torch.cat(all_arg, dim=0).unsqueeze(1)  # (B, 1, d)
        if len(all_union_axis) > 0:
            all_union_axis = torch.cat(all_union_axis, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_arg = torch.cat(all_union_arg, dim=0).unsqueeze(1)  # (B, 1, d)
            all_union_axis = all_union_axis.view(all_union_axis.shape[0] // 2, 2, 1, -1)
            all_union_arg = all_union_arg.view(all_union_arg.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_sample) != type(None):
            # 2.1 计算 一般的查询
            if len(all_axis) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]

                positive_axis = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_axis = self.angle_scale(positive_axis, self.axis_scale)
                positive_axis = convert_to_axis(positive_axis)

                positive_logit = self.cal_logit_cone(positive_axis, all_axis, all_arg)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            # 2.1 计算 并查询
            if len(all_union_axis) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]

                positive_axis = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_axis = self.angle_scale(positive_axis, self.axis_scale)
                positive_axis = convert_to_axis(positive_axis)

                positive_union_logit = self.cal_logit_cone(positive_axis, all_union_axis, all_union_arg)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        # 3. 计算负例损失
        if type(negative_sample) != type(None):
            # 3.1 计算 一般的查询
            if len(all_axis) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape

                negative_axis = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_axis = self.angle_scale(negative_axis, self.axis_scale)
                negative_axis = convert_to_axis(negative_axis)

                negative_logit = self.cal_logit_cone(negative_axis, all_axis, all_arg)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            # 3.1 计算 并查询
            if len(all_union_axis) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape

                negative_axis = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_axis = self.angle_scale(negative_axis, self.axis_scale)
                negative_axis = convert_to_axis(negative_axis)

                negative_union_logit = self.cal_logit_cone(negative_axis, all_union_axis, all_union_arg)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
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
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                if self.use_cuda:
                    arg_entity_embedding = arg_entity_embedding.cuda()

                idx += 1

                q_axis = axis_entity_embedding
                q_arg = arg_entity_embedding
            else:
                # 嵌入中间推理状态
                q_axis, q_arg, idx = self.embed_query_cone(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    q_axis, q_arg = self.cone_negation(q_axis, q_arg)

                # projection
                else:
                    r_axis = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    r_axis = self.angle_scale(r_axis, self.axis_scale)
                    r_axis = convert_to_axis(r_axis)

                    r_arg = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    r_arg = self.angle_scale(r_arg, self.arg_scale)
                    r_arg = convert_to_axis(r_arg)

                    q_axis, q_arg = self.cone_proj(q_axis, q_arg, r_axis, r_arg)
                idx += 1
        else:
            # 这一类比较复杂，表示最后一个运算是且运算
            #     (('e', ('r',)), ('e', ('r',))): '2i',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
            #     (('e', ('r', 'r')), ('e', ('r',))): 'pi',
            #     (('e', ('r',)), ('e', ('r', 'n'))): '2in',
            #     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
            #     (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
            #     (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
            # intersection
            axis_1_list = []
            arg_list = []
            for i in range(len(query_structure)):  # 把内部每个子结构都嵌入了，再执行 且运算
                q_axis, q_arg, idx = self.embed_query_cone(queries, query_structure[i], idx)
                axis_1_list.append(q_axis)
                arg_list.append(q_arg)

            stacked_axis_1 = torch.stack(axis_1_list)
            stacked_arg = torch.stack(arg_list)

            q_axis, q_arg = self.cone_intersection(stacked_axis_1, stacked_arg)

        return q_axis, q_arg, idx

    # implement distance function
    def distance(self, entity_axis, query_axis, query_arg):
        # inner distance
        distance2axis = torch.abs(torch.sin((entity_axis - query_axis) / 2))
        distance_base = torch.abs(torch.sin(query_arg / 2))
        distance_in = torch.min(distance2axis, distance_base)

        # outer distance
        delta1 = entity_axis - (query_axis - query_arg)
        delta2 = entity_axis - (query_axis + query_arg)
        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        return distance

    def cal_logit_cone(self, entity_axis, query_axis, query_arg):
        distance_1 = self.distance(entity_axis, query_axis, query_arg)
        logit = self.gamma - distance_1 * self.modulus
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

    def train_step(self, model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    def test_step(self, model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range.to(device))
                else:
                    if args.cuda:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda())
                    else:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1))

                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]

                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0

                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


if __name__ == "__main__":
    nentity, nrelation, hidden_dim = (5, 6, 3)
    model = SamE(nentity, nrelation, hidden_dim, 12, query_name_dict=query_name_dict, center_reg=0.02)
    positive_sample = torch.LongTensor(
        [
            [1, 2]
        ]
    )
    negative_sample = torch.LongTensor(
        [
            [1, 3],
            [2, 3],
        ]
    )
    subsampling_weight = 1
    batch_queries_dict = collections.defaultdict(list)
    batch_idxs_dict = collections.defaultdict(list)
    query = [1, 2]
    query_structure = ['e', 'r']
    batch_queries_dict[query_structure].append(query)
    batch_idxs_dict[query_structure].append(0)
    ans = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    print(ans.shape)
