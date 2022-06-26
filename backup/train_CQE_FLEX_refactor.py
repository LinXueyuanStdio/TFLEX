"""
@date: 2021/10/26
@description: 整理模型，重构代码，加了注释
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ComplexQueryData import *
from backup.dataloader import TestDataset, TrainDataset
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds
from backup.util import flatten_query


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * -1
    return y


class Projection(nn.Module):
    def __init__(self, dim, hidden_dim=1600, num_layers=2, drop=0.1):
        super(Projection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, q_feature, q_logic, r_feature, r_logic):
        x = torch.cat([q_feature + r_feature, q_logic + r_logic], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic = torch.chunk(x, 2, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        return feature, logic


class Intersection(nn.Module):
    def __init__(self, dim):
        super(Intersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic):
        # feature: N x B x d
        # logic:   N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic


class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + 2L) % (2L) - L, where L=1
        indicator_positive = feature >= 0
        indicator_negative = feature < 0
        feature[indicator_positive] = feature[indicator_positive] - 1
        feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    def forward(self, feature, logic):
        feature = self.neg_feature(feature)
        logic = 1 - logic
        return feature, logic


class FLEX(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 test_batch_size=1,
                 query_name_dict=None,
                 center_reg=None, drop: float = 0.):
        super(FLEX, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.projection = Projection(self.entity_dim, drop=drop)
        self.intersection = Intersection(self.entity_dim)
        self.negation = Negation()

        self.query_name_dict = query_name_dict
        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_FLEX(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_FLEX(self, positive_sample, negative_sample, subsampling_weight,
                     batch_queries_dict: Dict[QueryStructure, torch.Tensor],
                     batch_idxs_dict: Dict[QueryStructure, List[List[int]]]):
        # 1. 用 batch_queries_dict 将 查询 嵌入
        all_idxs, all_feature, all_logic = [], [], []
        all_union_idxs, all_union_feature, all_union_logic = [], [], []
        for query_structure in batch_queries_dict:
            # 用字典重新组织了嵌入，一个批次(BxL)只对应一种结构
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                feature, logic, _ = self.embed_query(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                                     self.transform_union_structure(query_structure),
                                                     0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_feature.append(feature)
                all_union_logic.append(logic)
            else:
                feature, logic, _ = self.embed_query(batch_queries_dict[query_structure],
                                                     query_structure,
                                                     0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_feature.append(feature)
                all_logic.append(logic)

        if len(all_feature) > 0:
            all_feature = torch.cat(all_feature, dim=0)  # (B, d)
            all_logic = torch.cat(all_logic, dim=0)  # (B, d)
        if len(all_union_feature) > 0:
            all_union_feature = torch.cat(all_union_feature, dim=0)  # (2B, d)
            all_union_logic = torch.cat(all_union_logic, dim=0)  # (2B, d)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_sample) != type(None):
            # 2.1 计算 一般的查询
            if len(all_feature) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]  # (B,)
                positive_feature = self.entity_feature(positive_sample_regular).unsqueeze(1)  # (B, 1, d)
                positive_scores = self.scoring_all(positive_feature, all_feature, all_logic)
            else:
                positive_scores = torch.Tensor([]).to(self.embedding_range.device)

            # 2.1 计算 并查询
            if len(all_union_feature) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]  # (B,)
                positive_feature = self.entity_feature(positive_sample_union).unsqueeze(1)  # (B, 1, d)
                positive_union_scores = self.scoring_all(positive_feature, all_union_feature, all_union_logic)

                batch_size = positive_union_scores.shape[0] // 2
                positive_union_scores = positive_union_scores.view(batch_size, 2, -1)
                positive_union_scores = torch.max(positive_union_scores, dim=1)[0]
                positive_union_scores = positive_union_scores.view(batch_size, -1)
            else:
                positive_union_scores = torch.Tensor([]).to(self.embedding_range.device)
            positive_scores = torch.cat([positive_scores, positive_union_scores], dim=0)
        else:
            positive_scores = None

        # 3. 计算负例损失
        if type(negative_sample) != type(None):
            # 3.1 计算 一般的查询
            if len(all_feature) > 0:
                negative_sample_regular = negative_sample[all_idxs]  # (B, N)
                negative_feature = self.entity_feature(negative_sample_regular)  # (B, N, d)
                negative_scores = self.scoring_all(negative_feature, all_feature, all_logic)
            else:
                negative_scores = torch.Tensor([]).to(self.embedding_range.device)

            # 3.1 计算 并查询
            if len(all_union_feature) > 0:
                negative_sample_union = negative_sample[all_union_idxs]  # (B, N)
                negative_feature = self.entity_feature(negative_sample_union).unsqueeze(1)  # (B, 1, N, d)
                batch_size = all_union_feature.shape[0] // 2
                all_union_feature = all_union_feature.view(batch_size, 2, 1, -1)
                all_union_logic = all_union_logic.view(batch_size, 2, 1, -1)
                negative_union_scores = self.scoring_all(negative_feature, all_union_feature, all_union_logic)

                negative_union_scores = negative_union_scores.view(batch_size, 2, -1)
                negative_union_scores = torch.max(negative_union_scores, dim=1)[0]
                negative_union_scores = negative_union_scores.view(batch_size, -1)
            else:
                negative_union_scores = torch.Tensor([]).to(self.embedding_range.device)
            negative_scores = torch.cat([negative_scores, negative_union_scores], dim=0)
        else:
            negative_scores = None

        return positive_scores, negative_scores, subsampling_weight, all_idxs + all_union_idxs

    def entity_feature(self, sample_ids):
        entity_feature = self.entity_feature_embedding(sample_ids)
        entity_feature = self.scale(entity_feature)
        entity_feature = convert_to_feature(entity_feature)
        return entity_feature

    def single_forward(self, query_structure: QueryStructure, queries: torch.Tensor, positive_sample: torch.Tensor, negative_sample: torch.Tensor):
        feature, logic, _ = self.embed_query(queries, query_structure, 0)
        positive_score = self.scoring_all(self.entity_feature(positive_sample), feature, logic)
        negative_score = self.scoring_all(self.entity_feature(negative_sample), feature, logic)
        return positive_score, negative_score

    def batch_forward(self, batch_queries_dict: Dict[QueryStructure, torch.Tensor], batch_idxs_dict: Dict[QueryStructure, List[List[int]]]):
        # 1. 用 batch_queries_dict 将查询嵌入（编码后的状态）
        all_idxs, all_feature, all_logic = [], [], []
        all_union_idxs, all_union_feature, all_union_logic = [], [], []
        for query_structure in batch_queries_dict:
            # 用字典重新组织了嵌入，一个批次(BxL)只对应一种结构
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                feature, logic, _ = self.embed_query(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                                     self.transform_union_structure(query_structure),
                                                     0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_feature.append(feature)
                all_union_logic.append(logic)
            else:
                feature, logic, _ = self.embed_query(batch_queries_dict[query_structure],
                                                     query_structure,
                                                     0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_feature.append(feature)
                all_logic.append(logic)

        if len(all_feature) > 0:
            all_feature = torch.cat(all_feature, dim=0)  # (B, d)
            all_logic = torch.cat(all_logic, dim=0)  # (B, d)
        if len(all_union_feature) > 0:
            all_union_feature = torch.cat(all_union_feature, dim=0)  # (2B, d)
            all_union_logic = torch.cat(all_union_logic, dim=0)  # (2B, d)

        # 1. 计算 一般的查询
        if len(all_feature) > 0:
            entity_feature = self.entity_feature_embedding.weight.unsqueeze(dim=0).repeat(all_feature.shape[0], 1, 1)  # (B, E, d)
            entity_feature = self.scale(entity_feature)
            entity_feature = convert_to_feature(entity_feature)
            negative_scores = self.scoring_all(entity_feature, all_feature, all_logic)
        else:
            negative_scores = torch.Tensor([]).to(feature.device)

        # 2. 计算 并查询
        if len(all_union_feature) > 0:
            entity_feature = self.entity_feature_embedding.weight.unsqueeze(dim=0).repeat(all_union_feature.shape[0], 1, 1)  # (2B, E, d)
            entity_feature = self.scale(entity_feature)
            entity_feature = convert_to_feature(entity_feature)
            negative_union_scores = self.scoring_all(entity_feature, all_union_feature, all_union_logic)

            batch_size = negative_union_scores.shape[0] // 2
            negative_union_scores = negative_union_scores.view(batch_size, 2, -1)
            negative_union_scores = torch.max(negative_union_scores, dim=1)[0]
            negative_union_scores = negative_union_scores.view(batch_size, -1)
        else:
            negative_union_scores = torch.Tensor([]).to(feature.device)
        negative_scores = torch.cat([negative_scores, negative_union_scores], dim=0)

        return negative_scores, all_idxs + all_union_idxs

    def distance(self, entity_feature, query_feature, query_logic):
        """
        entity_feature (B, E, d)
        query_feature  (B, 1, d)
        query_logic    (B, 1, d)
        query    =                 [(feature - logic) | feature | (feature + logic)]
        entity   = entity_feature            |             |               |
                         |                   |             |               |
        1) from entity to center of the interval           |               |
        d_center = entity_feature -                     feature            |
                         |<------------------------------->|               |
        2) from entity to left of the interval                             |
        d_left   = entity_feature - (feature - logic)                      |
                         |<----------------->|                             |
        3) from entity to right of the interval                            |
        d_right  = entity_feature -                               (feature + logic)
                         |<----------------------------------------------->|
        """
        d_center = entity_feature - query_feature
        d_left = entity_feature - (query_feature - query_logic)
        d_right = entity_feature - (query_feature + query_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, query_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < query_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring(self, entity_feature, query_feature, query_logic):
        distance = self.distance(entity_feature, query_feature, query_logic)
        score = self.gamma - distance * self.modulus
        return score

    def scoring_all(self, entity_feature, query_feature, query_logic):
        """
        entity_feature (B, E, d)
        feature        (B, d)
        logic          (B, d)
        """
        query_feature = query_feature.unsqueeze(dim=1)  # (B, 1, d)
        query_logic = query_logic.unsqueeze(dim=1)  # (B, 1, d)
        x = self.scoring(entity_feature, query_feature, query_logic)  # (B, E)
        return x

    def embed_query(self, queries: torch.Tensor, query_structure, idx: int):
        """
        迭代嵌入
        例子：(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
        B = 2, queries=[[1]]
        Iterative embed a batch of queries with same structure
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
                feature_entity_embedding = self.entity_feature(queries[:, idx])
                logic_entity_embedding = torch.zeros_like(feature_entity_embedding).to(feature_entity_embedding.device)

                idx += 1

                q_feature = feature_entity_embedding
                q_logic = logic_entity_embedding
            else:
                # 嵌入中间推理状态
                q_feature, q_logic, idx = self.embed_query(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    q_feature, q_logic = self.negation(q_feature, q_logic)

                # projection
                else:
                    r_feature = self.relation_feature_embedding(queries[:, idx])
                    r_feature = self.scale(r_feature)
                    r_feature = convert_to_feature(r_feature)

                    r_logic = self.relation_logic_embedding(queries[:, idx])
                    r_logic = self.scale(r_logic)
                    r_logic = convert_to_logic(r_logic)

                    q_feature, q_logic = self.projection(q_feature, q_logic, r_feature, r_logic)
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
            feature_list = []
            logic_list = []
            for i in range(len(query_structure)):  # 把内部每个子结构都嵌入了，再执行 且运算
                q_feature, q_logic, idx = self.embed_query(queries, query_structure[i], idx)
                feature_list.append(q_feature)
                logic_list.append(q_logic)

            stacked_feature = torch.stack(feature_list)
            stacked_logic = torch.stack(logic_list)

            q_feature, q_logic = self.intersection(stacked_feature, stacked_logic)

        return q_feature, q_logic, idx

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


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: ComplexQueryData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, tasks, evaluate_union, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg,
                 ):
        super(MyExperiment, self).__init__(output, local_rank=0)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        nentity = data.nentity
        nrelation = data.nrelation
        self.log('-------------------------------' * 3)
        self.log('# entity: %d' % nentity)
        self.log('# relation: %d' % nrelation)
        self.log('# max steps: %d' % max_steps)
        self.log('Evaluate unoins using: %s' % evaluate_union)

        self.log("loading data")

        # 1. build train dataset
        train_queries = data.train_queries
        train_answers = data.train_answers
        valid_queries = data.valid_queries
        valid_hard_answers = data.valid_hard_answers
        valid_easy_answers = data.valid_easy_answers
        test_queries = data.test_queries
        test_hard_answers = data.test_hard_answers
        test_easy_answers = data.test_easy_answers

        self.log("Training info:")
        for query_structure in train_queries:
            self.log(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, nentity, nrelation, negative_sample_size, train_answers),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, nentity, nrelation, negative_sample_size, train_answers),
                batch_size=batch_size,
                shuffle=True,
                num_workers=cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

        self.log("Validation info:")
        for query_structure in valid_queries:
            self.log(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(valid_queries, nentity, nrelation),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn
        )

        self.log("Test info:")
        for query_structure in test_queries:
            self.log(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(test_queries, nentity, nrelation),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn
        )

        # 2. build model
        model = FLEX(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            center_reg=center_reg,
            test_batch_size=test_batch_size,
            query_name_dict=query_name_dict,
            drop=input_dropout,
        ).to(train_device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_score = 0
        best_test_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            self.dump_model(model)
            model.eval()
            with torch.no_grad():
                self.debug("Resumed from score %.4f." % best_score)
                self.debug("Take a look at the performance after resumed.")
                self.debug("Validation (step: %d):" % start_step)
                result = self.evaluate(model, valid_easy_answers, valid_hard_answers, valid_dataloader, test_batch_size, test_device)
                best_score = self.visual_result(start_step + 1, result, "Valid")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate(model, test_easy_answers, test_hard_answers, test_dataloader, test_batch_size, test_device)
                best_test_score = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            self.dump_model(model)

        current_learning_rate = lr
        hyper = {
            'center_reg': center_reg,
            'tasks': tasks,
            'learning_rate': lr,
            'batch_size': batch_size,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
        }
        self.metric_log_store.add_hyper(hyper)
        for k, v in hyper.items():
            self.log(f'{k} = {v}')
        self.metric_log_store.add_progress(max_steps)
        warm_up_steps = max_steps // 2

        # 3. training
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            log = self.train(model, opt, train_path_iterator, step, train_device)
            for metric in log:
                self.vis.add_scalar('path_' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = self.train(model, opt, train_other_iterator, step, train_device)
                for metric in log:
                    self.vis.add_scalar('other_' + metric, log[metric], step)
                log = self.train(model, opt, train_path_iterator, step, train_device)

            progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("positive", log["positive_sample_loss"]), ("negative", log["negative_sample_loss"])])
            if (step + 1) % 10 == 0:
                self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                print("")
                self.log('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_easy_answers, valid_hard_answers, valid_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Valid")
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.debug("saving best score %.4f" % score)
                        self.metric_log_store.add_best_metric({"result": result}, "Valid")
                        self.model_param_store.save_best(model, opt, step, 0, score)
                    else:
                        self.model_param_store.save_by_score(model, opt, step, 0, score)
                        self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_easy_answers, test_hard_answers, test_dataloader, test_batch_size, test_device)
                    score = self.visual_result(step + 1, result, "Test")
                    if score >= best_test_score:
                        best_test_score = score
                        self.metric_log_store.add_best_metric({"result": result}, "Test")
                    print("")
        self.metric_log_store.finish()

    def train(self, model, optimizer, train_iterator, step, device="cuda:0"):
        model.train()
        model.to(device)
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict: Dict[List[str], list] = collections.defaultdict(list)
        batch_idxs_dict: Dict[List[str], List[int]] = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)
        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)

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

    def evaluate(self, model, easy_answers, hard_answers, test_dataloader, test_batch_size, device="cuda:0"):
        model.to(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = collections.defaultdict(list)
        step = 0
        h10 = None
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for negative_sample, queries, queries_unflatten, query_structures in test_dataloader:
            batch_queries_dict.clear()
            batch_idxs_dict.clear()
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)

            negative_logit, idxs = model.batch_forward(batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.float()
            if len(argsort) == test_batch_size:
                # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                ranking = ranking.scatter_(1, argsort, model.batch_entity_range.to(device))  # achieve the ranking of all entities
            else:
                # otherwise, create a new torch Tensor for batch_entity_range
                ranking = ranking.scatter_(1,
                                           argsort,
                                           torch.arange(model.nentity).float().repeat(argsort.shape[0], 1).to(device)
                                           )  # achieve the ranking of all entities
            for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                hard_answer = hard_answers[query]
                easy_answer = easy_answers[query]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy
                answer_list = torch.arange(num_hard + num_easy).float().to(device)
                cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                mrr = torch.mean(1. / cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).float()).item()
                h3 = torch.mean((cur_ranking <= 3).float()).item()
                h10 = torch.mean((cur_ranking <= 10).float()).item()
                query_structure_name = query_name_dict[query_structure]
                logs[query_structure_name].append({
                    'MRR': mrr,
                    'hits@1': h1,
                    'hits@3': h3,
                    'hits@10': h10,
                    'hard': num_hard,
                })

            step += 1
            progbar.update(step, [("Hits @10", h10)])

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure_name in logs:
            for metric in logs[query_structure_name][0].keys():
                if metric in ['hard']:
                    continue
                metrics[query_structure_name][metric] = sum([log[metric] for log in logs[query_structure_name]]) / len(logs[query_structure_name])
            metrics[query_structure_name]['num_queries'] = len(logs[query_structure_name])

        return metrics

    def visual_result(self, step_num: int, result, scope: str):
        """Evaluate queries in dataloader"""
        self.metric_log_store.add_metric({scope: result}, step_num, scope)
        average_metrics = defaultdict(float)
        num_query_structures = 0
        num_queries = 0
        for query_structure in result:
            for metric in result[query_structure]:
                self.vis.add_scalar("_".join([scope, query_structure, metric]), result[query_structure][metric], step_num)
                if metric != 'num_queries':
                    average_metrics[metric] += result[query_structure][metric]
            num_queries += result[query_structure]['num_queries']
            num_query_structures += 1

        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            self.vis.add_scalar("_".join([scope, 'average', metric]), average_metrics[metric], step_num)

        header = "{0:<8s}".format(scope)
        row_results = defaultdict(list)
        row_results[header].append("avg")
        row_results["num_queries"].append(num_queries)
        for row in average_metrics:
            cell = average_metrics[row]
            row_results[row].append(cell)
        for col in result:
            row_results[header].append(col)
            col_data = result[col]
            for row in col_data:
                cell = col_data[row]
                row_results[row].append(cell)

        def to_str(data):
            if isinstance(data, float):
                return "{0:>6.2%}  ".format(data)
            elif isinstance(data, int):
                return "{0:^6d}  ".format(data)
            else:
                return "{0:^6s}  ".format(data)

        for i in row_results:
            row = row_results[i]
            self.log("{0:<8s}".format(i)[:8] + ": " + "".join([to_str(data) for data in row]))
        score = average_metrics["MRR"]
        return score


@click.command()
@click.option("--data_home", type=str, default="data/reasoning", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, NELL.")
@click.option("--name", type=str, default="FLEX_base", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=300001, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10000, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=10000, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=8, help="Test batch size.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option('--tasks', type=str, default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
@click.option('--evaluate_union', type=str, default="DNF", help='choices=[DNF, DM] the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
@click.option('--cpu_num', type=int, default=4, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=30.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, tasks, evaluate_union, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg,
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)

    if dataset == "FB15k-237":
        dataset = FB15k_237_BetaE(data_home)
    elif dataset == "FB15k":
        dataset = FB15k_BetaE(data_home)
    elif dataset == "NELL":
        dataset = NELL_BetaE(data_home)
    cache = ComplexQueryDatasetCachePath(dataset.root_path)
    data = ComplexQueryData(cache_path=cache)
    data.load(evaluate_union, tasks)

    MyExperiment(
        output, data,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, tasks, evaluate_union, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg,
    )


if __name__ == '__main__':
    main()
