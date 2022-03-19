"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/26
@description: null
"""
from collections import defaultdict
from typing import List, Dict, Tuple

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import expression
from ComplexTemporalQueryData import ICEWS05_15, ICEWS14, ComplexTemporalQueryDatasetCachePath, ComplexQueryData, TYPE_train_queries_answers
from ComplexTemporalQueryDataloader import TestDataset, TrainDataset
from expression.ParamSchema import is_entity, is_relation, is_timestamp
from expression.TFLEX_DSL import union_query_structures
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds

QueryStructure = Tuple[str, List[str]]


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * -1
    return y


def convert_to_time_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * -1
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
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

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
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, time_density = torch.chunk(x, 5, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        time_density = convert_to_time_density(time_density)
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
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        time_density = convert_to_time_density(time_density)
        return feature, logic, time_feature, time_logic, time_density


class Intersection(nn.Module):
    def __init__(self, dim):
        super(Intersection, self).__init__()
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
    def __init__(self):
        super(TemporalBefore, self).__init__()

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


class TemporalAfter(nn.Module):
    def __init__(self):
        super(TemporalAfter, self).__init__()

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


class TemporalNext(nn.Module):
    def __init__(self):
        super(TemporalNext, self).__init__()

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


class Union(nn.Module):
    def __init__(self, dim):
        super(Union, self).__init__()
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
        time_logic, _ = torch.max(time_logic, dim=0)
        time_density, _ = torch.max(time_density, dim=0)
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

        logic, _ = torch.max(logic, dim=0)
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
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.timestamp_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)

        self.timestamp_time_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)

        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_density_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.intersection = Intersection(hidden_dim)
        self.union = Union(hidden_dim)
        self.negation = Negation()

        self.time_projection = TimeProjection(hidden_dim, drop=drop)
        self.time_intersection = TemporalIntersection(hidden_dim)
        self.time_union = TemporalUnion(hidden_dim)
        self.time_negation = TemporalNegation()

        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg

        def And(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.intersection(feature, logic, time_feature, time_logic, time_density)

        def And3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic, q3_time_density = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density, q3_time_density])
            return self.intersection(feature, logic, time_feature, time_logic, time_density)

        def Or(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_time_density = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_time_density = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            time_density = torch.stack([q1_time_density, q2_time_density])
            return self.union(feature, logic, time_feature, time_logic, time_density)

        def Not(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.negation(feature, logic, time_feature, time_logic, time_density)

        def TimeNot(q):
            feature, logic, time_feature, time_logic, time_density = q
            return self.time_negation(feature, logic, time_feature, time_logic, time_density)

        def EntityProjection2(s, r, t):
            s_feature, s_logic, s_time_feature, s_time_logic, s_time_density = s
            r_feature, r_logic, r_time_feature, r_time_logic, r_time_density = r
            t_feature, t_logic, t_time_feature, t_time_logic, t_time_density = t
            return self.entity_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_time_density,
                r_feature, r_logic, r_time_feature, r_time_logic, r_time_density,
                t_feature, t_logic, t_time_feature, t_time_logic, t_time_density
            )

        def TimeProjection2(s, r, o):
            s_feature, s_logic, s_time_feature, s_time_logic, s_time_density = s
            r_feature, r_logic, r_time_feature, r_time_logic, r_time_density = r
            o_feature, o_logic, o_time_feature, o_time_logic, o_time_density = o
            return self.entity_projection(
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
        self.parser = expression.NeuralParser(neural_ops)

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.timestamp_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_density_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    def entity_token(self, idx):
        feature = convert_to_feature(self.scale(self.entity_feature_embedding(idx)))
        logic = torch.zeros_like(feature).to(feature.device)
        time_feature = torch.zeros_like(feature).to(feature.device)
        time_logic = torch.zeros_like(feature).to(feature.device)
        time_density = torch.zeros_like(feature).to(feature.device)
        return feature, logic, time_feature, time_logic, time_density

    def relation_token(self, idx):
        feature = convert_to_feature(self.scale(self.relation_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.relation_logic_embedding(idx)))
        time_feature = convert_to_time_feature(self.scale(self.relation_time_feature_embedding(idx)))
        time_logic = convert_to_time_logic(self.scale(self.relation_time_logic_embedding(idx)))
        time_density = convert_to_time_density(self.scale(self.relation_time_density_embedding(idx)))
        return feature, logic, time_feature, time_logic, time_density

    def timestamp_token(self, idx):
        time_feature = convert_to_time_feature(self.scale(self.timestamp_time_feature_embedding(idx)))
        feature = torch.zeros_like(time_feature).to(time_feature.device)
        logic = torch.zeros_like(feature).to(feature.device)
        time_logic = torch.zeros_like(feature).to(feature.device)
        time_density = torch.zeros_like(feature).to(feature.device)
        return feature, logic, time_feature, time_logic, time_density

    def embed_args(self, query_args, query_tensor):
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
        return embedding_of_args

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_FLEX(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_FLEX(self, positive_sample, negative_sample, subsampling_weight,
                     batch_queries_dict: Dict[QueryStructure, torch.Tensor],
                     batch_idxs_dict: Dict[QueryStructure, List[List[int]]]):
        # 1. 用 batch_queries_dict 将 查询 嵌入
        all_idxs, all_feature, all_logic = [], [], []
        all_union_idxs, all_union_feature, all_union_logic = [], [], []

        for query_structure in batch_queries_dict:
            query_name, query_args = query_structure
            query_tensor = batch_queries_dict[query_structure]  # BxL, B for batch size, L for query args length
            query_idxs = batch_idxs_dict[query_structure]
            # query_idxs is of shape Bx1.
            # each element indicates global index of each row in query_tensor.
            # global index means the index in sample from dataloader.
            # the sample is grouped by query name and leads to query_tensor here.
            if query_name in union_query_structures:
                # transform to DNF
                func = self.parser.fast_function(query_name + "_DNF")
                embedding_of_args = self.embed_args(query_args, query_tensor)
                predict = func(*embedding_of_args) # tuple[B x dt, B x dt]
                all_union_idxs.extend(query_idxs)
            else:
                # DM remains
                func = self.parser.fast_function(query_name)
                embedding_of_args = self.embed_args(query_args, query_tensor) # [B x dt]*L
                predict = func(*embedding_of_args) # B x dt
                all_idxs.extend(query_idxs)

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
            pass

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

    def embed_query(self, queries: torch.Tensor, query_structure):
        """
        迭代嵌入
        例子：(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
        B = 2, queries=[[1]]
        Iterative embed a batch of queries with same structure
        queries:(B, L): a flattened batch of queries (all queries are of query_structure), B is batch size, L is length of queries
        """
        pass


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: ComplexQueryData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, tasks, evaluate_union, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg,
                 ):
        super(MyExperiment, self).__init__(output)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        entity_count = data.entity_count
        relation_count = data.relation_count
        timestamp_count = data.timestamp_count
        self.log('-------------------------------' * 3)
        self.log('# entity: %d' % entity_count)
        self.log('# relation: %d' % relation_count)
        self.log('# timestamp: %d' % timestamp_count)
        self.log('# max steps: %d' % max_steps)

        # 1. build train dataset
        train_queries_answers = data.train_queries_answers
        valid_queries_answers = data.valid_queries_answers
        test_queries_answers = data.test_queries_answers

        self.log("Training info:")
        for query_structure_name in train_queries_answers:
            self.log(query_structure_name + ": " + str(len(train_queries_answers[query_structure_name]["queries_answers"])))
        train_path_queries: TYPE_train_queries_answers = {}
        # List[Tuple[List[int], Set[int]]]
        train_other_queries: TYPE_train_queries_answers = {}
        path_list = ["Pe", "Pt", "Pe2", 'Pe3']
        for query_structure_name in train_queries_answers:
            if query_structure_name in path_list:
                train_path_queries[query_structure_name] = train_queries_answers[query_structure_name]
            else:
                train_other_queries[query_structure_name] = train_queries_answers[query_structure_name]
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, entity_count, relation_count, negative_sample_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, entity_count, relation_count, negative_sample_size),
                batch_size=batch_size,
                shuffle=True,
                num_workers=cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

        self.log("Validation info:")
        for query_structure_name in valid_queries_answers:
            self.log(query_structure_name + ": " + str(len(valid_queries_answers[query_structure_name]["queries_answers"])))
        valid_dataloader = DataLoader(
            TestDataset(valid_queries_answers, entity_count, relation_count),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn
        )

        self.log("Test info:")
        for query_structure_name in test_queries_answers:
            self.log(query_structure_name + ": " + str(len(test_queries_answers[query_structure_name]["queries_answers"])))
        test_dataloader = DataLoader(
            TestDataset(test_queries_answers, entity_count, relation_count),
            batch_size=test_batch_size,
            num_workers=cpu_num // 2,
            collate_fn=TestDataset.collate_fn
        )

        # 2. build model
        model = FLEX(
            nentity=entity_count,
            nrelation=relation_count,
            ntimestamp=timestamp_count,
            hidden_dim=hidden_dim,
            gamma=gamma,
            center_reg=center_reg,
            test_batch_size=test_batch_size,
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
                result = self.evaluate(model, valid_dataloader, test_batch_size, test_device)
                best_score = self.visual_result(start_step + 1, result, "Valid")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate(model, test_dataloader, test_batch_size, test_device)
                best_test_score = self.visual_result(start_step + 1, result, "Test")
        else:
            model.init()
            self.dump_model(model)

        current_learning_rate = lr
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
                    result = self.evaluate(model, valid_dataloader, test_batch_size, test_device)
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
                    result = self.evaluate(model, test_dataloader, test_batch_size, test_device)
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

        query_name, args, batch_queries, positive_answer, negative_answer, subsampling_weight = next(train_iterator)
        batch_queries_dict: Dict[Tuple[str, List[str]], list] = defaultdict(list)
        batch_idxs_dict: Dict[Tuple[str, List[str]], List[int]] = defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            query_schema = (query_name[i], args[i])
            batch_queries_dict[query_schema].append(query)
            batch_idxs_dict[query_schema].append(i)
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)
        positive_answer = positive_answer.to(device)
        negative_answer = negative_answer.to(device)
        subsampling_weight = subsampling_weight.to(device)

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_answer, negative_answer, subsampling_weight, batch_queries_dict, batch_idxs_dict)

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

    def evaluate(self, model, test_dataloader, test_batch_size, device="cuda:0"):
        model.to(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(list)
        step = 0
        h10 = None
        batch_queries_dict = defaultdict(list)
        batch_idxs_dict = defaultdict(list)
        for candidate_answer, query_name, args, batch_queries, easy_answer, hard_answer in test_dataloader:
            batch_queries_dict.clear()
            batch_idxs_dict.clear()
            for i, query in enumerate(batch_queries):
                query_schema = (query_name[i], args[i])
                batch_queries_dict[query_schema].append(query)
                batch_idxs_dict[query_schema].append(i)
            for query_schema in batch_queries_dict:
                batch_queries_dict[query_schema] = torch.LongTensor(batch_queries_dict[query_schema]).to(device)
            candidate_answer = candidate_answer.to(device)

            _, negative_logit, _, idxs = model(None, candidate_answer, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [query_name[i] for i in idxs]
            easy_answer_reorder = [easy_answer[i] for i in idxs]
            hard_answer_reorder = [hard_answer[i] for i in idxs]
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
            for idx, (i, query_structure_name, easy_answer, hard_answer) in enumerate(zip(argsort[:, 0], queries_unflatten, easy_answer_reorder, hard_answer_reorder)):
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
                logs[query_structure_name].append({
                    'MRR': mrr,
                    'hits@1': h1,
                    'hits@3': h3,
                    'hits@10': h10,
                    'hard': num_hard,
                })

            step += 1
            progbar.update(step, [("Hits @10", h10)])

        metrics = defaultdict(lambda: defaultdict(int))
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

    if dataset == "ICEWS14":
        dataset = ICEWS14(data_home)
    elif dataset == "ICEWS05_15":
        dataset = ICEWS05_15(data_home)
    cache = ComplexTemporalQueryDatasetCachePath(dataset.root_path)
    data = ComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
        "train_queries_answers", "valid_queries_answers", "test_queries_answers",
    ])

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
