"""
@date: 2021/10/26
@description: null
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
from ComplexQueryData import QueryStructure, query_name_dict
from backup.dataloader import TestDataset, TrainDataset
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds
from backup.util import flatten_query

pi = 3.14159265358979323846


def convert_to_arg(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_axis(x):
    # [-1,1]
    y = x * 1
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = 1
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

        nn.init.xavier_uniform_(self.axis_layer_1.weight)
        nn.init.xavier_uniform_(self.axis_layer_2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis, arg):
        # axis: N x B x d
        # arg:  N x B x d
        logits = torch.cat([axis, arg], dim=-1)  # N x B x 2d

        axis_attention = F.softmax(self.axis_layer_2(F.relu(self.axis_layer_1(logits))), dim=0)
        axis = torch.sum(axis_attention * axis, dim=0)

        arg = torch.prod(arg, dim=0)
        return axis, arg


class Negation(nn.Module):
    def __init__(self, dim):
        super(Negation, self).__init__()
        self.dim = dim
        self.arg_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.arg_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.arg_layer_1.weight)
        nn.init.xavier_uniform_(self.arg_layer_2.weight)

    def forward(self, axis, arg):
        logits = torch.cat([axis, arg], dim=-1)  # N x B x 2d
        axis = self.arg_layer_2(F.relu(self.arg_layer_1(logits)))
        arg = 1 - arg
        return axis, arg


class PercentagE(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 test_batch_size=1,
                 use_cuda=False,
                 query_name_dict=None,
                 center_reg=None, drop: float = 0.):
        super(PercentagE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        torch.autograd.set_detect_anomaly(True)
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
        self.cone_negation = Negation(self.entity_dim)

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
            all_union_axis = torch.cat(all_union_axis, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_arg = torch.cat(all_union_arg, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_axis = all_union_axis.view(all_union_axis.shape[0] // 2, 2, 1, -1)  # (B, 2, 1, d)
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
        # inner distance 这里 sin(x) 近似为 L1 范数
        distance2axis = torch.abs(entity_axis - query_axis)
        distance_base = torch.abs(query_arg)
        distance_in = torch.min(distance2axis, distance_base)

        # outer distance
        delta1 = entity_axis - (query_axis - query_arg)
        delta2 = entity_axis - (query_axis + query_arg)
        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(delta1), torch.abs(delta2))
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


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: ComplexQueryData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
                 negative_sample_size, hidden_dim, gamma, learning_rate, cpu_num,
                 tasks, evaluate_union, center_reg,
                 ):
        super(MyExperiment, self).__init__(output, local_rank=0)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts(["train_CQE_BoolE_mlpNegation.py"])
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
        model = PercentagE(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            use_cuda=True,
            center_reg=center_reg,
            test_batch_size=test_batch_size,
            query_name_dict=query_name_dict,
            drop=input_dropout,
        ).to(train_device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_score = 0
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
                self.visual_result(start_step + 1, result, "Valid")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate(model, test_easy_answers, test_hard_answers, test_dataloader, test_batch_size, test_device)
                self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            self.dump_model(model)

        current_learning_rate = learning_rate
        self.log('center reg = %s' % center_reg)
        self.log('tasks = %s' % tasks)
        self.log('start_step = %d' % start_step)
        self.log('Start Training...')
        self.log('learning_rate = %f' % current_learning_rate)
        self.log('batch_size = %d' % batch_size)
        self.log('hidden_dim = %d' % hidden_dim)
        self.log('gamma = %f' % gamma)
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
            if (step + 1) % 1000 == 0:
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
                    self.visual_result(step + 1, result, "Test")
                    print("")

    def train(self, model, optimizer, train_iterator, step, device="cuda:0"):
        model.train()
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
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps // (test_batch_size * 10))
        logs = collections.defaultdict(list)
        step = 0
        h10 = None
        for negative_sample, queries, queries_unflatten, query_structures in test_dataloader:
            batch_queries_dict = collections.defaultdict(list)
            batch_idxs_dict = collections.defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)
            negative_sample = negative_sample.to(device)

            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().float()
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

                logs[query_structure].append({
                    'MRR': mrr,
                    'hits@1': h1,
                    'hits@3': h3,
                    'hits@10': h10,
                    'hard': num_hard,
                })

            step += 1
            if step % (test_batch_size * 10) == 0:
                progbar.update(step // (test_batch_size * 10), [("Hits @10", h10)])

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['hard']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics

    def visual_result(self, step_num: int, result, scope: str):
        """Evaluate queries in dataloader"""
        average_metrics = defaultdict(float)
        num_query_structures = 0
        num_queries = 0
        for query_structure in result:
            for metric in result[query_structure]:
                self.vis.add_scalar("_".join([scope, query_name_dict[query_structure], metric]), result[query_structure][metric], step_num)
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
            row_results[header].append(query_name_dict[col])
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
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="BoolE_mlpNegation", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=300001, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10000, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=10000, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=4, help="Test batch size.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option("--hidden_dropout1", type=float, default=0.2, help="Dropout after the first hidden layer.")
@click.option("--hidden_dropout2", type=float, default=0.2, help="Dropout after the second hidden layer.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option('--hidden_dim', default=800, type=int, help="embedding dimension")
@click.option('--gamma', default=30.0, type=float, help="margin in the loss")
@click.option('--learning_rate', default=0.0001, type=float)
@click.option('--cpu_num', default=4, type=int, help="used to speed up torch.dataloader")
@click.option('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
@click.option('--center_reg', default=0.02, type=float, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--evaluate_union', default="DNF", type=str,
              help='choices=[DNF, DM] the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
def main(dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
         negative_sample_size, hidden_dim, gamma, learning_rate, cpu_num,
         tasks, evaluate_union, center_reg,
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)

    dataset = FB15k_237_BetaE()
    cache = ComplexQueryDatasetCachePath(dataset.root_path)
    data = ComplexQueryData(cache_path=cache)
    data.load(evaluate_union, tasks)

    MyExperiment(
        output, data,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, sampling_window_size, label_smoothing,
        train_device, test_device,
        resume, resume_by_score,
        lr, amsgrad, lr_decay, weight_decay,
        edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
        negative_sample_size, hidden_dim, gamma, learning_rate, cpu_num,
        tasks, evaluate_union, center_reg,
    )


if __name__ == '__main__':
    main()
