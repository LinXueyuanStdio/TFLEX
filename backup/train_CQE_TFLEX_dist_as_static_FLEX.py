"""
@date: 2021/10/26
@description: TODO 忽略时间信息，视为静态图进行补全
"""
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Set

import click
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import expression
from ComplexTemporalQueryData import ICEWS05_15, ICEWS14, ComplexTemporalQueryDatasetCachePath, ComplexQueryData, TYPE_train_queries_answers
from ComplexTemporalQueryDataloader import TestDataset, TrainDataset
from expression.ParamSchema import is_entity, is_relation, is_timestamp
from expression.TFLEX_DSL import is_to_predict_entity_set, query_contains_union_and_we_should_use_DNF, test_query_structures
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds

QueryStructure = str
TYPE_token = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TYPE_query_token = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TYPE_relation_token = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TYPE_timestamp_token = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# 下面 3 行代码解决 tensor 在不同进程中共享问题。
# 共享是通过读写文件的，如果同时打开文件的进程数太多，会崩。
# 这里是把允许的最大进程数设为 4096，比较大了。
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

L = 1


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_time_density(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


class EntityProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(EntityProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 5
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic, q_time_feature, q_time_logic, q_time_density,
                r_feature, r_logic, r_time_feature, r_time_logic, r_time_density,
                t_feature, t_logic, t_time_feature, t_time_logic, t_time_density):
        x = torch.cat([
            q_feature + r_feature + t_feature,
            q_logic + r_logic + t_logic,
            q_time_feature + r_time_feature + t_time_feature,
            q_time_logic + r_time_logic + t_time_logic,
            q_time_density + r_time_density + t_time_density,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, time_density = torch.chunk(x, 5, dim=-1)
        return feature, logic, time_feature, time_logic, time_density


class TimeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(TimeProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 5
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self,
                q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density,
                r_feature, r_logic, r_time_feature, r_time_logic, r_time_density,
                q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density):
        x = torch.cat([
            q1_feature + r_feature + q2_feature,
            q1_logic + r_logic + q2_logic,
            q1_time_feature + r_time_feature + q2_time_feature,
            q1_time_logic + r_time_logic + q2_time_logic,
            q1_time_density + r_time_density + q2_time_density,
        ], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, time_density = torch.chunk(x, 5, dim=-1)
        # feature = convert_to_feature(feature)
        # logic = convert_to_logic(logic)
        # time_feature = convert_to_time_feature(time_feature)
        # time_logic = convert_to_time_logic(time_logic)
        # time_density = convert_to_time_density(time_density)
        return feature, logic, time_feature, time_logic, time_density


class EntityIntersection(nn.Module):
    def __init__(self, dim):
        super(EntityIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        time_density, _ = torch.min(time_density, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic, time_density


class TemporalIntersection(nn.Module):
    def __init__(self, dim):
        super(TemporalIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        time_density, _ = torch.min(time_density, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic, time_density


class EntityNegation(nn.Module):
    def __init__(self):
        super(EntityNegation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + 2L) % (2L) - L, where L=1
        indicator_positive = feature >= 0
        indicator_negative = feature < 0
        feature[indicator_positive] = feature[indicator_positive] - 1
        feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        feature = self.neg_feature(feature)
        logic = 1 - logic
        return feature, logic, time_feature, time_logic, time_density


class TemporalNegation(nn.Module):
    def __init__(self):
        super(TemporalNegation, self).__init__()

    def neg_feature(self, feature):
        # f,f' in [-L, L]
        # f' = (f + 2L) % (2L) - L, where L=1
        indicator_positive = feature >= 0
        indicator_negative = feature < 0
        feature[indicator_positive] = feature[indicator_positive] - 1
        feature[indicator_negative] = feature[indicator_negative] + 1
        return feature

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        time_feature = self.neg_feature(time_feature)
        time_logic = 1 - time_logic
        return feature, logic, time_feature, time_logic, time_density


class TemporalBefore(nn.Module):
    def __init__(self, dim):
        super(TemporalBefore, self).__init__()
        self.dim = dim
        self.time_density_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_density_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.time_density_layer_1.weight)
        nn.init.xavier_uniform_(self.time_density_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # theta_x_left = theta_x - theta_y / 2
        theta_left = time_feature - time_logic / 2
        center = theta_left + L
        time_feature = center / 2
        time_logic = center

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 3d
        time_density = convert_to_time_density(self.time_density_layer_2(F.relu(self.time_density_layer_1(logits))))
        # time_density = convert_to_time_density (time_logic * time_density) / center
        return feature, logic, time_feature, time_logic, time_density


class TemporalAfter(nn.Module):
    def __init__(self, dim):
        super(TemporalAfter, self).__init__()
        self.dim = dim
        self.time_density_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_density_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.time_density_layer_1.weight)
        nn.init.xavier_uniform_(self.time_density_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # theta_x_right = theta_x + theta_y / 2
        theta_right = time_feature + time_logic / 2
        center = L - theta_right
        time_feature = center / 2
        time_logic = center

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 3d
        time_density = convert_to_time_density(self.time_density_layer_2(F.relu(self.time_density_layer_1(logits))))
        # time_density = (time_logic * time_density) / center
        return feature, logic, time_feature, time_logic, time_density


class TemporalNext(nn.Module):
    def __init__(self, timestamp_dim, get_timestamps_delta, get_timestamps_origin):
        super(TemporalNext, self).__init__()
        self.timestamp_dim = timestamp_dim
        self.get_timestamps_origin = get_timestamps_origin
        self.get_timestamps_delta = get_timestamps_delta

    def next_feature(self, time_feature):
        timestamp_delta = self.get_timestamps_delta()
        if len(time_feature.shape) == 2:
            time_feature = time_feature + timestamp_delta
        else:
            time_feature = time_feature + timestamp_delta.unsqueeze(dim=0)
        time_feature = convert_to_time_feature(time_feature)
        return time_feature

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        time_feature = self.next_feature(time_feature)
        return feature, logic, time_feature, time_logic, time_density


class EntityUnion(nn.Module):
    def __init__(self, dim):
        super(EntityUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.max(logic, dim=0)
        # for time, it is intersection
        time_logic, _ = torch.min(time_logic, dim=0)
        time_density, _ = torch.min(time_density, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic, time_density


class TemporalUnion(nn.Module):
    def __init__(self, dim):
        super(TemporalUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 3, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, time_density):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic, time_density], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        # for entity, it is intersection
        logic, _ = torch.min(logic, dim=0)
        # for time, it is union
        time_logic, _ = torch.max(time_logic, dim=0)
        time_density, _ = torch.max(time_density, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic, time_density


class FLEX(nn.Module):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(FLEX, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntimestamp = ntimestamp
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.timestamp_dim = hidden_dim
        print(locals())

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)
        self.entity_logic_embedding = nn.Embedding(nentity, self.entity_dim)
        self.entity_time_feature_embedding = nn.Embedding(nentity, self.entity_dim)
        self.entity_time_logic_embedding = nn.Embedding(nentity, self.entity_dim)
        self.entity_time_density_embedding = nn.Embedding(nentity, self.entity_dim)

        self.timestamp_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)
        self.timestamp_logic_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)
        self.timestamp_time_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)
        self.timestamp_time_logic_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)
        self.timestamp_time_density_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)

        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_density_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation()

        self.time_projection = TimeProjection(hidden_dim, drop=drop)
        self.time_intersection = TemporalIntersection(hidden_dim)
        self.time_union = TemporalUnion(hidden_dim)
        self.time_negation = TemporalNegation()
        self.time_before = TemporalBefore(hidden_dim)
        self.time_after = TemporalAfter(hidden_dim)
        self.time_next = TemporalNext(hidden_dim, get_timestamps_delta=lambda: self.timestamp_delta.detach(), get_timestamps_origin=lambda: self.timestamp_origin.detach())

        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg
        self.parser = self.build_parser()

    def build_parser(self):
        def And(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.entity_intersection(feature, logic, time_feature, time_logic, time_density)

        def And3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic, q3_time_density = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density, q3_time_density])
            return self.entity_intersection(feature, logic, time_feature, time_logic, time_density)

        def Or(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.entity_union(feature, logic, time_feature, time_logic, time_density)

        def Not(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.entity_negation(feature, logic, time_feature, time_logic, time_density)

        def TimeNot(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.time_negation(feature, logic, time_feature, time_logic, time_density)

        def EntityProjection2(e1, r1, t1):
            s_feature, s_logic, s_time_feature, s_time_logic, s_time_density = e1
            r_feature, r_logic, r_time_feature, r_time_logic, r_time_density = r1
            t_feature, t_logic, t_time_feature, t_time_logic, t_time_density = t1
            return self.entity_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_time_density,
                r_feature, r_logic, r_time_feature, r_time_logic, r_time_density,
                t_feature, t_logic, t_time_feature, t_time_logic, t_time_density
            )

        def TimeProjection2(e1, r1, e2):
            s_feature, s_logic, s_time_feature, s_time_logic, s_time_density = e1
            r_feature, r_logic, r_time_feature, r_time_logic, r_time_density = r1
            o_feature, o_logic, o_time_feature, o_time_logic, o_time_density = e2
            return self.time_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_time_density,
                r_feature, r_logic, r_time_feature, r_time_logic, r_time_density,
                o_feature, o_logic, o_time_feature, o_time_logic, o_time_density
            )

        def TimeAnd(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.time_intersection(feature, logic, time_feature, time_logic, time_density)

        def TimeAnd3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic, q3_time_density = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density, q3_time_density])
            return self.time_intersection(feature, logic, time_feature, time_logic, time_density)

        def TimeOr(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.time_union(feature, logic, time_feature, time_logic, time_density)

        def TimeBefore(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.time_before(feature, logic, time_feature, time_logic, time_density)

        def TimeAfter(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.time_after(feature, logic, time_feature, time_logic, time_density)

        def TimeNext(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.time_next(feature, logic, time_feature, time_logic, time_density)

        neural_ops = {
            "And": And,
            "And3": And3,
            "Or": Or,
            "Not": Not,
            "EntityProjection": EntityProjection2,
            "TimeProjection": TimeProjection2,
            "TimeAnd": TimeAnd,
            "TimeAnd3": TimeAnd3,
            "TimeOr": TimeOr,
            "TimeNot": TimeNot,
            "TimeBefore": TimeBefore,
            "TimeAfter": TimeAfter,
            "TimeNext": TimeNext,
        }
        return expression.NeuralParser(neural_ops)

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.entity_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.entity_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.entity_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.entity_time_density_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.timestamp_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.timestamp_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.timestamp_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.timestamp_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.timestamp_time_density_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_density_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    def entity_feature(self, idx):
        # return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))
        return self.scale(self.entity_feature_embedding(idx))

    def timestamp_feature(self, idx):
        # B = idx.shape[0]
        # feature = self.timestamp_origin + torch.mm(idx.view(-1, 1).float(), self.timestamp_delta)
        # if len(idx.shape) == 1:
        #     feature = feature.view(B, self.timestamp_dim)
        # else:
        #     feature = feature.view(B, -1, self.timestamp_dim)
        return self.scale(self.timestamp_time_feature_embedding(idx))
        # return convert_to_time_feature(self.scale(feature))

    def entity_token(self, idx) -> TYPE_token:
        feature = self.scale(self.entity_feature_embedding(idx))
        logic = self.scale(self.entity_logic_embedding(idx))
        time_feature = self.scale(self.entity_time_feature_embedding(idx))
        time_logic = self.scale(self.entity_time_logic_embedding(idx))
        time_density = self.scale(self.entity_time_density_embedding(idx))
        return feature, logic, time_feature, time_logic, time_density

    def relation_token(self, idx) -> TYPE_token:
        feature = self.scale(self.relation_feature_embedding(idx))
        logic = self.scale(self.relation_logic_embedding(idx))
        time_feature = self.scale(self.relation_time_feature_embedding(idx))
        time_logic = self.scale(self.relation_time_logic_embedding(idx))
        time_density = self.scale(self.relation_time_density_embedding(idx))
        return feature, logic, time_feature, time_logic, time_density

    def timestamp_token(self, idx) -> TYPE_token:
        feature = self.scale(self.timestamp_feature_embedding(idx))
        logic = self.scale(self.timestamp_logic_embedding(idx))
        time_feature = self.scale(self.timestamp_time_feature_embedding(idx))
        time_logic = self.scale(self.timestamp_time_logic_embedding(idx))
        time_density = self.scale(self.timestamp_time_density_embedding(idx))
        return feature, logic, time_feature, time_logic, time_density

    def embed_args(self, query_args: List[str], query_tensor: torch.Tensor) -> TYPE_token:
        embedding_of_args = []
        for i in range(len(query_args)):
            arg_name = query_args[i]
            tensor = query_tensor[:, i]
            if is_entity(arg_name):
                token_embedding = self.entity_token(tensor)
            elif is_relation(arg_name):
                token_embedding = self.relation_token(tensor)
            elif is_timestamp(arg_name):
                token_embedding = self.timestamp_token(tensor)
            else:
                raise Exception("Unknown Args %s" % arg_name)
            embedding_of_args.append(token_embedding)
        return tuple(embedding_of_args)

    def cat_to_tensor(self, token_list: List[TYPE_token]) -> TYPE_token:
        feature = []
        logic = []
        time_feature = []
        time_logic = []
        time_density = []
        for x in token_list:
            feature.append(x[0])
            logic.append(x[1])
            time_feature.append(x[2])
            time_logic.append(x[3])
            time_density.append(x[4])
        feature = torch.cat(feature, dim=0).unsqueeze(1)
        logic = torch.cat(logic, dim=0).unsqueeze(1)
        time_feature = torch.cat(time_feature, dim=0).unsqueeze(1)
        time_logic = torch.cat(time_logic, dim=0).unsqueeze(1)
        time_density = torch.cat(time_density, dim=0).unsqueeze(1)
        return feature, logic, time_feature, time_logic, time_density

    def forward(self,
                train_data_list: Optional[List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
                test_data_list: Optional[List[Tuple[str, torch.Tensor, torch.Tensor]]]):
        if train_data_list is not None:
            return self.forward_train(train_data_list)
        elif test_data_list is not None:
            return self.forward_test(test_data_list)
        else:
            raise Exception("Train or Test, please choose one!")

    def forward_train(self, data_list: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        positive_scores, negative_scores, subsampling_weights = [], [], []
        for query_structure, query, positive_answer, negative_answer, subsampling_weight in data_list:
            positive_score, negative_score = self.forward_predict_2(query_structure, query, positive_answer, negative_answer)
            positive_scores.append(positive_score)
            negative_scores.append(negative_score)
            subsampling_weights.append(subsampling_weight)
        positive_scores = torch.cat(positive_scores, dim=0)
        negative_scores = torch.cat(negative_scores, dim=0)
        subsampling_weights = torch.cat(subsampling_weights, dim=0)
        return positive_scores, negative_scores, subsampling_weights

    def forward_test(self, data_list: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> Dict[QueryStructure, torch.Tensor]:
        """
        return {"Pe": (B, L) }
        L 是答案个数，[预测实体]和[预测时间戳]的答案个数不一样，所以不能对齐合并
        不同结构的 L 不同
        一般用于valid/test，不用于train
        """
        grouped_score = {}

        for query_structure, query, answer in data_list:
            # query (B, L), B for batch size, L for query args length
            # answer (B, N)
            grouped_score[query_structure] = self.forward_predict(query_structure, query, answer)

        return grouped_score

    def forward_FLEX(self,
                     positive_answer: Optional[torch.Tensor],
                     negative_answer: Optional[torch.Tensor],
                     subsampling_weight: Optional[torch.Tensor],
                     grouped_query: Dict[QueryStructure, torch.Tensor],
                     grouped_idxs: Dict[QueryStructure, List[List[int]]]):
        """
        positive_answer: None or (B, )
        negative_answer: None or (B, N)
        subsampling_weight: None or (B, )
        """
        # 1. 将 查询 嵌入到低维空间
        (all_idxs_e, all_predict_e), \
        (all_idxs_t, all_predict_t), \
        (all_union_idxs_e, all_union_predict_e), \
        (all_union_idxs_t, all_union_predict_t) = self.batch_predict(grouped_query, grouped_idxs)

        all_idxs = all_idxs_e + all_idxs_t + all_union_idxs_e + all_union_idxs_t
        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs]

        positive_scores = None
        negative_scores = None

        # 2. 计算正例损失
        if positive_answer is not None:
            scores_e = self.scoring_to_answers_by_idxs(all_idxs_e, positive_answer, all_predict_e, predict_entity=True, DNF_predict=False)
            scores_t = self.scoring_to_answers_by_idxs(all_idxs_t, positive_answer, all_predict_t, predict_entity=False, DNF_predict=False)
            scores_union_e = self.scoring_to_answers_by_idxs(all_union_idxs_e, positive_answer, all_union_predict_e, predict_entity=True, DNF_predict=True)
            scores_union_t = self.scoring_to_answers_by_idxs(all_union_idxs_t, positive_answer, all_union_predict_t, predict_entity=False, DNF_predict=True)
            positive_scores = torch.cat([scores_e, scores_t, scores_union_e, scores_union_t], dim=0)

        # 3. 计算负例损失
        if negative_answer is not None:
            scores_e = self.scoring_to_answers_by_idxs(all_idxs_e, negative_answer, all_predict_e, predict_entity=True, DNF_predict=False)
            scores_t = self.scoring_to_answers_by_idxs(all_idxs_t, negative_answer, all_predict_t, predict_entity=False, DNF_predict=False)
            scores_union_e = self.scoring_to_answers_by_idxs(all_union_idxs_e, negative_answer, all_union_predict_e, predict_entity=True, DNF_predict=True)
            scores_union_t = self.scoring_to_answers_by_idxs(all_union_idxs_t, negative_answer, all_union_predict_t, predict_entity=False, DNF_predict=True)
            negative_scores = torch.cat([scores_e, scores_t, scores_union_e, scores_union_t], dim=0)

        return positive_scores, negative_scores, subsampling_weight

    def single_predict(self, query_structure: QueryStructure, query_tensor: torch.Tensor) -> Union[TYPE_token, Tuple[TYPE_token, TYPE_token]]:
        query_name, query_args = query_structure
        if query_contains_union_and_we_should_use_DNF(query_name):
            # transform to DNF
            func = self.parser.fast_function(query_name + "_DNF")
            embedding_of_args = self.embed_args(query_args, query_tensor)
            predict_1, predict_2 = func(*embedding_of_args)  # tuple[B x dt, B x dt]
            return predict_1, predict_2
        else:
            # other query and DM are normal
            func = self.parser.fast_function(query_name)
            embedding_of_args = self.embed_args(query_args, query_tensor)  # [B x dt]*L
            predict = func(*embedding_of_args)  # B x dt
            return predict

    def batch_predict(self, grouped_query: Dict[QueryStructure, torch.Tensor], grouped_idxs: Dict[QueryStructure, List[List[int]]]):
        all_idxs_e, all_predict_e = [], []
        all_idxs_t, all_predict_t = [], []
        all_union_idxs_e, all_union_predict_1_e, all_union_predict_2_e = [], [], []
        all_union_idxs_t, all_union_predict_1_t, all_union_predict_2_t = [], [], []
        all_union_predict_e: Optional[TYPE_token] = None
        all_union_predict_t: Optional[TYPE_token] = None

        for query_structure in grouped_query:
            query_name = query_structure
            query_args = self.parser.fast_args(query_name)
            query_tensor = grouped_query[query_structure]  # (B, L), B for batch size, L for query args length
            query_idxs = grouped_idxs[query_structure]
            # query_idxs is of shape Bx1.
            # each element indicates global index of each row in query_tensor.
            # global index means the index in sample from dataloader.
            # the sample is grouped by query name and leads to query_tensor here.
            if query_contains_union_and_we_should_use_DNF(query_name):
                # transform to DNF
                func = self.parser.fast_function(query_name + "_DNF")
                embedding_of_args = self.embed_args(query_args, query_tensor)
                predict_1, predict_2 = func(*embedding_of_args)  # tuple[(B, d), (B, d)]
                if is_to_predict_entity_set(query_name):
                    all_union_predict_1_e.append(predict_1)
                    all_union_predict_2_e.append(predict_2)
                    all_union_idxs_e.extend(query_idxs)
                else:
                    all_union_predict_1_t.append(predict_1)
                    all_union_predict_2_t.append(predict_2)
                    all_union_idxs_t.extend(query_idxs)
            else:
                # other query and DM are normal
                func = self.parser.fast_function(query_name)
                embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
                predict = func(*embedding_of_args)  # (B, d)
                if is_to_predict_entity_set(query_name):
                    all_predict_e.append(predict)
                    all_idxs_e.extend(query_idxs)
                else:
                    all_predict_t.append(predict)
                    all_idxs_t.extend(query_idxs)

        def cat_to_tensor(token_list: List[TYPE_token]) -> TYPE_token:
            feature = []
            logic = []
            time_feature = []
            time_logic = []
            time_density = []
            for x in token_list:
                feature.append(x[0])
                logic.append(x[1])
                time_feature.append(x[2])
                time_logic.append(x[3])
                time_density.append(x[4])
            feature = torch.cat(feature, dim=0).unsqueeze(1)
            logic = torch.cat(logic, dim=0).unsqueeze(1)
            time_feature = torch.cat(time_feature, dim=0).unsqueeze(1)
            time_logic = torch.cat(time_logic, dim=0).unsqueeze(1)
            time_density = torch.cat(time_density, dim=0).unsqueeze(1)
            return feature, logic, time_feature, time_logic, time_density

        if len(all_idxs_e) > 0:
            all_predict_e = cat_to_tensor(all_predict_e)  # (B, 1, d) * 5
        if len(all_idxs_t) > 0:
            all_predict_t = cat_to_tensor(all_predict_t)  # (B, 1, d) * 5
        if len(all_union_idxs_e) > 0:
            all_union_predict_1_e = cat_to_tensor(all_union_predict_1_e)  # (B, 1, d) * 5
            all_union_predict_2_e = cat_to_tensor(all_union_predict_2_e)  # (B, 1, d) * 5
            all_union_predict_e: TYPE_token = tuple([torch.cat([x, y], dim=1) for x, y in zip(all_union_predict_1_e, all_union_predict_2_e)])  # (B, 2, d) * 5
        if len(all_union_idxs_t) > 0:
            all_union_predict_1_t = cat_to_tensor(all_union_predict_1_t)  # (B, 1, d) * 5
            all_union_predict_2_t = cat_to_tensor(all_union_predict_2_t)  # (B, 1, d) * 5
            all_union_predict_t: TYPE_token = tuple([torch.cat([x, y], dim=1) for x, y in zip(all_union_predict_1_t, all_union_predict_2_t)])  # (B, 2, d) * 5
        return (all_idxs_e, all_predict_e), \
               (all_idxs_t, all_predict_t), \
               (all_union_idxs_e, all_union_predict_e), \
               (all_union_idxs_t, all_union_predict_t)

    def grouped_predict(self, grouped_query: Dict[QueryStructure, torch.Tensor], grouped_answer: Dict[QueryStructure, torch.Tensor]) -> Dict[QueryStructure, torch.Tensor]:
        """
        return {"Pe": (B, L) }
        L 是答案个数，预测实体和预测时间戳 的答案个数不一样，所以不能对齐合并
        不同结构的 L 不同
        一般用于valid/test，不用于train
        """
        grouped_score = {}

        for query_structure in grouped_query:
            query = grouped_query[query_structure]  # (B, L), B for batch size, L for query args length
            answer = grouped_answer[query_structure]  # (B, N)
            grouped_score[query_structure] = self.forward_predict(query_structure, query, answer)

        return grouped_score

    def forward_predict(self, query_structure: QueryStructure, query_tensor: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        # query_tensor  # (B, L), B for batch size, L for query args length
        # answer  # (B, N)
        query_name = query_structure
        query_args = self.parser.fast_args(query_name)
        # the sample is grouped by query name and leads to query_tensor here.
        if query_contains_union_and_we_should_use_DNF(query_name):
            # transform to DNF
            func = self.parser.fast_function(query_name + "_DNF")
            embedding_of_args = self.embed_args(query_args, query_tensor)
            predict_1, predict_2 = func(*embedding_of_args)  # tuple[(B, d), (B, d)]
            all_union_predict: TYPE_token = tuple([torch.stack([x, y], dim=1).unsqueeze(dim=2) for x, y in zip(predict_1, predict_2)])  # (B, 1, 1, dt) or (B, 2, 1, dt)
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(answer, all_union_predict, predict_entity=True, DNF_predict=True)
            else:
                return self.scoring_to_answers(answer, all_union_predict, predict_entity=False, DNF_predict=True)
        else:
            # other query and DM are normal
            func = self.parser.fast_function(query_name)
            embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
            predict = func(*embedding_of_args)  # (B, d)
            all_predict: TYPE_token = tuple([i.unsqueeze(dim=1).unsqueeze(dim=1) for i in predict])  # (B, 1, 1, dt) or (B, 2, 1, dt)
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(answer, all_predict, predict_entity=True, DNF_predict=False)
            else:
                return self.scoring_to_answers(answer, all_predict, predict_entity=False, DNF_predict=False)

    def forward_predict_2(self, query_structure: QueryStructure, query_tensor: torch.Tensor,
                          positive_answer: torch.Tensor, negative_answer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query_tensor  # (B, L), B for batch size, L for query args length
        # answer  # (B, N)
        query_name = query_structure
        query_args = self.parser.fast_args(query_name)
        # the sample is grouped by query name and leads to query_tensor here.
        if query_contains_union_and_we_should_use_DNF(query_name):
            # transform to DNF
            func = self.parser.fast_function(query_name + "_DNF")
            embedding_of_args = self.embed_args(query_args, query_tensor)
            predict_1, predict_2 = func(*embedding_of_args)  # tuple[(B, d), (B, d)]
            all_union_predict: TYPE_token = tuple([torch.stack([x, y], dim=1).unsqueeze(dim=2) for x, y in zip(predict_1, predict_2)])  # (B, 1, 1, dt) or (B, 2, 1, dt)
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(positive_answer, all_union_predict, predict_entity=True, DNF_predict=True), \
                       self.scoring_to_answers(negative_answer, all_union_predict, predict_entity=True, DNF_predict=True)
            else:
                return self.scoring_to_answers(positive_answer, all_union_predict, predict_entity=False, DNF_predict=True), \
                       self.scoring_to_answers(negative_answer, all_union_predict, predict_entity=False, DNF_predict=True)
        else:
            # other query and DM are normal
            func = self.parser.fast_function(query_name)
            embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
            predict = func(*embedding_of_args)  # (B, d)
            all_predict: TYPE_token = tuple([i.unsqueeze(dim=1).unsqueeze(dim=1) for i in predict])  # (B, 1, 1, dt) or (B, 2, 1, dt)
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(positive_answer, all_predict, predict_entity=True, DNF_predict=False), \
                       self.scoring_to_answers(negative_answer, all_predict, predict_entity=True, DNF_predict=False)
            else:
                return self.scoring_to_answers(positive_answer, all_predict, predict_entity=False, DNF_predict=False), \
                       self.scoring_to_answers(negative_answer, all_predict, predict_entity=False, DNF_predict=False)

    def scoring_to_answers_by_idxs(self, all_idxs, answer: torch.Tensor, q: TYPE_token, predict_entity=True, DNF_predict=False):
        """
        B for batch size
        N for negative sampling size (maybe N=1 when positive samples only)
        all_answer_idxs: (B, ) or (B, N) int
        all_predict:     (B, 1, dt) or (B, 2, dt) float
        return score:    (B, N) float
        """
        if len(all_idxs) <= 0:
            return torch.Tensor([]).to(self.embedding_range.device)
        answer_ids = answer[all_idxs]
        answer_ids = answer_ids.view(answer_ids.shape[0], -1)
        return self.scoring_to_answers(answer_ids, q, predict_entity, DNF_predict)

    def scoring_to_answers(self, answer_ids: torch.Tensor, q: TYPE_token, predict_entity=True, DNF_predict=False):
        """
        B for batch size
        N for negative sampling size (maybe N=1 when positive samples only)
        answer_ids:   (B, N) int
        all_predict:  (B, 1, 1, dt) or (B, 2, 1, dt) float
        return score: (B, N) float
        """
        if predict_entity:
            feature, logic, time_feature, time_logic, time_density = self.entity_token(answer_ids)
            feature = feature.unsqueeze(dim=1)  # (B, 1, N, d)
            logic = logic.unsqueeze(dim=1)  # (B, 1, N, d)
            time_feature = time_feature.unsqueeze(dim=1)  # (B, 1, N, d)
            time_logic = time_logic.unsqueeze(dim=1)  # (B, 1, N, d)
            time_density = time_density.unsqueeze(dim=1)  # (B, 1, N, d)
            a = feature, logic, time_feature, time_logic, time_density
            scores = self.scoring_entity(a, q)  # (B, 1, N) or (B, 2, N)
        else:
            feature, logic, time_feature, time_logic, time_density = self.timestamp_token(answer_ids)
            feature = feature.unsqueeze(dim=1)  # (B, 1, N, d)
            logic = logic.unsqueeze(dim=1)  # (B, 1, N, d)
            time_feature = time_feature.unsqueeze(dim=1)  # (B, 1, N, d)
            time_logic = time_logic.unsqueeze(dim=1)  # (B, 1, N, d)
            time_density = time_density.unsqueeze(dim=1)  # (B, 1, N, d)
            a = feature, logic, time_feature, time_logic, time_density
            scores = self.scoring_timestamp(a, q)  # (B, 1, N) or (B, 2, N)

        if DNF_predict:
            scores = torch.max(scores, dim=1)[0]  # (B, N)
        else:
            scores = scores.squeeze(dim=1)  # (B, N)
        return scores  # (B, N)

    def distance_between_entity_and_query(self, entity_feature, query_feature, query_logic):
        """
        entity_feature (B, 1, N, d)
        query_feature  (B, 1, 1, dt) or (B, 2, 1, dt)
        query_logic    (B, 1, 1, dt) or (B, 2, 1, dt)
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

    def distance_between_timestamp_and_query(self, timestamp_feature, time_feature, time_logic, time_density):
        """
        entity_feature (B, 1, N, d)
        query_feature  (B, 1, 1, dt) or (B, 2, 1, dt)
        query_logic    (B, 1, 1, dt) or (B, 2, 1, dt)
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
        d_center = timestamp_feature - time_feature
        d_left = timestamp_feature - (time_feature - time_logic)
        d_right = timestamp_feature - (time_feature + time_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, time_logic) * time_density
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right)) * time_density
        outer_distance[feature_distance < time_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring_entity(self, entity_token: TYPE_token, q: TYPE_token):
        e_feature, e_logic, e_time_feature, e_time_logic, e_time_density = entity_token
        feature, logic, time_feature, time_logic, time_density = q
        distance = self.distance_between_entity_and_query(e_feature, feature, logic)
        score = self.gamma - distance * self.modulus
        return score

    def scoring_timestamp(self, timestamp_token, q: TYPE_token):
        feature, logic, time_feature, time_logic, time_density = q
        distance = self.distance_between_timestamp_and_query(timestamp_token, time_feature, time_logic, time_density)
        score = self.gamma - distance * self.modulus
        return score


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: ComplexQueryData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg, local_rank
                 ):
        super(MyExperiment, self).__init__(output, local_rank)
        if local_rank == 0:
            self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        entity_count = data.entity_count
        relation_count = data.relation_count
        timestamp_count = data.timestamp_count
        max_relation_id = relation_count
        nprocs = torch.cuda.device_count()
        self.nprocs = nprocs
        world_size = nprocs  # TODO 暂时用单机多卡。多机多卡需要重新设置
        self.world_size = world_size
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        # batch_size = batch_size // nprocs
        # test_batch_size = test_batch_size // nprocs
        if local_rank == 0:
            self.log('-------------------------------' * 3)
            self.log('# entity: %d' % entity_count)
            self.log('# relation: %d' % relation_count)
            self.log('# timestamp: %d' % timestamp_count)
            self.log('# max steps: %d' % max_steps)

        # 1. build train dataset
        train_queries_answers = data.train_queries_answers
        valid_queries_answers = data.valid_queries_answers
        test_queries_answers = data.test_queries_answers

        train_path_queries: TYPE_train_queries_answers = {}
        train_other_queries: TYPE_train_queries_answers = {}
        path_list = ["Pe", "Pt", "Pe2", 'Pe3']
        for query_structure_name in train_queries_answers:
            if query_structure_name in path_list:
                train_path_queries[query_structure_name] = train_queries_answers[query_structure_name]
            else:
                train_other_queries[query_structure_name] = train_queries_answers[query_structure_name]
        train_path_dataset = TrainDataset(train_path_queries, entity_count, timestamp_count, negative_sample_size)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            train_path_dataset,
            sampler=DistributedSampler(train_path_dataset),
            batch_size=batch_size,
            # shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn2
        ))
        if len(train_other_queries) > 0:
            train_other_dataset = TrainDataset(train_other_queries, entity_count, timestamp_count, negative_sample_size)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                train_other_dataset,
                sampler=DistributedSampler(train_other_dataset),
                batch_size=batch_size,
                # shuffle=True,
                num_workers=cpu_num,
                collate_fn=TrainDataset.collate_fn2
            ))
        else:
            train_other_iterator = None

        valid_dataset = TestDataset(valid_queries_answers, entity_count, timestamp_count)
        valid_dataloader = DataLoader(
            valid_dataset,
            sampler=DistributedSampler(valid_dataset),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn2
        )

        test_dataset = TestDataset(test_queries_answers, entity_count, timestamp_count)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=DistributedSampler(test_dataset),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn2
        )
        if local_rank == 0:
            self.log("Training info:")
            for query_structure_name in train_queries_answers:
                self.log(query_structure_name + ": " + str(len(train_queries_answers[query_structure_name]["queries_answers"])))
            self.log("Validation info:")
            for query_structure_name in valid_queries_answers:
                self.log(query_structure_name + ": " + str(len(valid_queries_answers[query_structure_name]["queries_answers"])))
            self.log("Test info:")
            for query_structure_name in test_queries_answers:
                self.log(query_structure_name + ": " + str(len(test_queries_answers[query_structure_name]["queries_answers"])))

        # 2. build model
        model = FLEX(
            nentity=entity_count,
            nrelation=relation_count + max_relation_id,  # with reverse relations
            ntimestamp=timestamp_count,
            hidden_dim=hidden_dim,
            gamma=gamma,
            center_reg=center_reg,
            test_batch_size=test_batch_size,
            drop=input_dropout,
        ).cuda(local_rank)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_score = 0
        best_test_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            if local_rank == 0:
                self.dump_model(model)
                model.eval()
                with torch.no_grad():
                    self.debug("Resumed from score %.4f." % best_score)
                    self.debug("Take a look at the performance after resumed.")
                    self.debug("Validation (step: %d):" % start_step)
                    result = self.evaluate(model, valid_dataloader, local_rank)
                    best_score, _ = self.visual_result(start_step + 1, result, "Valid")
                    self.debug("Test (step: %d):" % start_step)
                    result = self.evaluate(model, test_dataloader, local_rank)
                    best_test_score, _ = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            if local_rank == 0:
                self.dump_model(model)
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

        current_learning_rate = lr
        if local_rank == 0:
            hyper = {
                'center_reg': center_reg,
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
            log = self.train(model, opt, train_path_iterator, step, local_rank)
            for metric in log:
                self.vis.add_scalar('path_' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = self.train(model, opt, train_other_iterator, step, local_rank)
                for metric in log:
                    self.vis.add_scalar('other_' + metric, log[metric], step)
                log = self.train(model, opt, train_path_iterator, step, local_rank)

            if local_rank == 0:
                progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("positive", log["positive_sample_loss"]), ("negative", log["negative_sample_loss"])])
            if (step + 1) % 10 == 0:
                self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                if local_rank == 0:
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
                    if local_rank == 0:
                        print("")
                        self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_dataloader, local_rank)
                    if local_rank == 0:
                        score, row_results = self.visual_result(step + 1, result, "Valid")
                        if score >= best_score:
                            self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                            best_score = score
                            self.metric_log_store.add_best_metric({"result": result}, "Valid")
                            self.debug("saving best score %.4f" % score)
                            self.model_param_store.save_best(model, opt, step, 0, score)
                            self.latex_store.save_best_valid_result(row_results)
                        else:
                            self.model_param_store.save_by_score(model, opt, step, 0, score)
                            self.latex_store.save_valid_result_by_score(row_results, score)
                            self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    if local_rank == 0:
                        print("")
                        self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_dataloader, local_rank)
                    if local_rank == 0:
                        score, row_results = self.visual_result(step + 1, result, "Test")
                        self.latex_store.save_test_result_by_score(row_results, score)
                        if score >= best_test_score:
                            best_test_score = score
                            self.latex_store.save_best_test_result(row_results)
                            self.metric_log_store.add_best_metric({"result": result}, "Test")
                        print("")
        if local_rank == 0:
            self.metric_log_store.finish()

    def train(self, model, optimizer, train_iterator, step, device="cuda:0"):
        model.train()
        model.cuda(device)

        data_list = next(train_iterator)
        cuda_data_list = []
        for query_name, query_tensor, positive_answer, negative_answer, subsampling_weight in data_list:
            query_tensor = query_tensor.cuda(device, non_blocking=True)
            positive_answer = positive_answer.cuda(device, non_blocking=True)
            negative_answer = negative_answer.cuda(device, non_blocking=True)
            subsampling_weight = subsampling_weight.cuda(device, non_blocking=True)
            cuda_data_list.append((query_name, query_tensor, positive_answer, negative_answer, subsampling_weight))
        # if device == 0:
        #     print()
        #     print("cuda_data_list", len(cuda_data_list), sum([i.shape[0] for _, i, _, _, _ in cuda_data_list]))
        #     print()

        positive_logit, negative_logit, subsampling_weight = model(cuda_data_list, None)

        negative_sample_loss = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_sample_loss = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_sample_loss).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_sample_loss).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        torch.distributed.barrier()
        log = {
            'positive_sample_loss': reduce_mean(positive_sample_loss, self.nprocs).item(),
            'negative_sample_loss': reduce_mean(negative_sample_loss, self.nprocs).item(),
            'loss': reduce_mean(loss, self.nprocs).item(),
        }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return log

    def evaluate(self, model, test_dataloader, device="cuda:0"):
        model.cuda(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(list)
        step = 0
        mrr = None
        h10 = None
        for data_list, grouped_easy_answer, grouped_hard_answer in test_dataloader:
            cuda_data_list = []
            for query_name, query_tensor, candidate_answer in data_list:
                query_tensor = query_tensor.cuda(device, non_blocking=True)
                candidate_answer = candidate_answer.cuda(device, non_blocking=True)
                cuda_data_list.append((query_name, query_tensor, candidate_answer))

            grouped_score = model(None, cuda_data_list)
            for query_name in grouped_score:
                score = grouped_score[query_name]
                easy_answer_mask: List[torch.Tensor] = grouped_easy_answer[query_name]
                hard_answer: List[Set[int]] = grouped_hard_answer[query_name]
                score[easy_answer_mask] = -float('inf')  # we remove easy answer, because easy answer may exist in training set
                ranking = score.argsort(dim=1, descending=True).cpu()  # sorted idx (B, N)

                ranks = []
                hits = []
                for i in range(10):
                    hits.append([])
                num_queries = ranking.shape[0]
                for i in range(num_queries):
                    for answer_id in hard_answer[i]:
                        rank = torch.where(ranking[i] == answer_id)[0][0]
                        ranks.append(1 / (rank + 1))
                        for hits_level in range(10):
                            hits[hits_level].append(1.0 if rank <= hits_level else 0.0)
                mrr = torch.mean(torch.FloatTensor(ranks)).item()
                h1 = torch.mean(torch.FloatTensor(hits[0])).item()
                h3 = torch.mean(torch.FloatTensor(hits[2])).item()
                h10 = torch.mean(torch.FloatTensor(hits[9])).item()
                logs[query_name].append({
                    'MRR': ranks,
                    'hits@1': hits[0],
                    'hits@2': hits[1],
                    'hits@3': hits[2],
                    'hits@5': hits[4],
                    'hits@10': hits[9],
                    'num_queries': num_queries,
                })

            if device == 0:
                step += 1
                progbar.update(step, [("MRR", mrr), ("Hits @10", h10)])

        # sync and reduce
        # 分布式评估 很麻烦，多任务学习的分布式评估更麻烦，因为每个进程采样到的任务数不一样
        # 下面这坨就是在绝对视野里强行对齐所有进程的任务结果，进行 reduce，最后汇总给 rank=0 的 master node 展示到命令行
        torch.distributed.barrier()  # 所有进程运行到这里时，都等待一下，直到所有进程都在这行，然后一起同时往后分别运行
        query_name_keys = []
        metric_name_keys = []
        all_tensors = []
        metric_names = list(logs[list(logs.keys())[0]][0].keys())
        for query_name in test_query_structures:  # test_query_structures 内是所有任务
            for metric_name in metric_names:
                query_name_keys.append(query_name)
                metric_name_keys.append(metric_name)
                if query_name in logs:
                    if metric_name == "num_queries":
                        value = sum([log[metric_name] for log in logs[query_name]])
                        all_tensors.append([value, 1])
                    else:
                        values = []
                        for log in logs[query_name]:
                            values.extend(log[metric_name])
                        all_tensors.append([sum(values), len(values)])
                else:
                    all_tensors.append([0, 1])
        all_tensors = torch.FloatTensor(all_tensors).to(device)
        metrics = defaultdict(lambda: defaultdict(float))
        dist.reduce(all_tensors, dst=0)  # 再次同步，每个进程都分享自己的 all_tensors 给其他进程，每个进程都看到所有数据了
        # 每个进程看到的数据都是所有数据的和，如果有 n 个进程，则有 all_tensors = 0.all_tensors + 1.all_tensors + ... + n.all_tensors
        if dist.get_rank() == 0:  # 我们只在 master 节点上处理，其他进程的结果丢弃了
            # 1. store to dict
            m = defaultdict(lambda: defaultdict(lambda x:[0, 1]))
            for query_name, metric, value in zip(query_name_keys, metric_name_keys, all_tensors):
                m[query_name][metric] = value
            # 2. delete empty query
            del_query = []
            for query_name in m:
                if m[query_name]["num_queries"][0] == 0:
                    del_query.append(query_name)
            for query_name in del_query:
                del m[query_name]
                query_name_keys.remove(query_name)
            # 3. correct values
            for query_name, metric in zip(query_name_keys, metric_name_keys):
                if query_name not in m:
                    continue
                value = m[query_name][metric]
                if metric == "num_queries":
                    metrics[query_name][metric] = int(value[0])
                else:
                    metrics[query_name][metric] = value[0] / value[1]

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
        for col in sorted(result):
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
        return score, row_results


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15.")
@click.option("--name", type=str, default="TFLEX_base", help="Name of the experiment.")
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
@click.option('--cpu_num', type=int, default=4, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=30.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--local_rank', type=int, default=-1, help='node rank for distributed training')
@click.option('--tasks', type=str, default="Pe,Pt", help='tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task')
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg, local_rank, tasks
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)

    if dataset == "ICEWS14":
        dataset = ICEWS14(data_home)
    elif dataset == "ICEWS05_15":
        dataset = ICEWS05_15(data_home)
    cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
    data = ComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
        "train_queries_answers", "valid_queries_answers", "test_queries_answers",
    ])
    tasks = tasks.split(",")
    for query_name in set(data.train_queries_answers.keys()) - set(tasks):
        data.train_queries_answers.pop(query_name)
    for query_name in set(data.valid_queries_answers.keys()) - set(tasks):
        data.valid_queries_answers.pop(query_name)
    for query_name in set(data.test_queries_answers.keys()) - set(tasks):
        data.test_queries_answers.pop(query_name)

    MyExperiment(
        output, data,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg, local_rank
    )


if __name__ == '__main__':
    main()
    # max_id = 20
    # entity_count = max_id
    # relation_count = max_id
    # timestamp_count = max_id
    # hidden_dim = 10
    # gamma = 10
    # center_reg = 0.02
    # test_batch_size = 1
    # input_dropout = 0.1
    # model = FLEX(
    #     nentity=entity_count,
    #     nrelation=relation_count,
    #     ntimestamp=timestamp_count,
    #     hidden_dim=hidden_dim,
    #     gamma=gamma,
    #     center_reg=center_reg,
    #     test_batch_size=test_batch_size,
    #     drop=input_dropout,
    # )
    # B = 8
    # query_args = ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "t3"]
    # query_structure = ("Pe_e2u", query_args)
    # query_tensor = torch.randint(0, max_id, (B, len(query_args)))
    # predict = model.single_predict(query_structure, query_tensor)
    # print(predict)
