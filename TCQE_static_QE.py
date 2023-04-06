"""
@date: 2021/10/26
@description: null
"""
from typing import Callable, List, Dict, Tuple, Optional, Union

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

import expression
from ComplexTemporalQueryData import ICEWS05_15, ICEWS14, ComplexTemporalQueryDatasetCachePath, ComplexQueryData, GDELT
from expression.ParamSchema import is_entity, is_relation, is_timestamp
from expression.TFLEX_DSL import is_to_predict_entity_set, query_contains_union_and_we_should_use_DNF
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.RandomSeeds import set_seeds
from train_TCQE_TFLEX import MyExperiment


QueryStructure = str
TYPE_token = Tuple[torch.Tensor, torch.Tensor]

pi = 3.14159265358979323846


def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


def convert_to_logic(x):
    return convert_to_arg(x)


def convert_to_feature(x):
    return convert_to_axis(x)


def convert_to_time_feature(x):
    return convert_to_axis(x)


class EntityProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(EntityProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 2
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic,
                r_feature, r_logic,
                t_feature, t_logic):
        x = torch.cat([
            q_feature + r_feature,
            q_logic + r_logic,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic = torch.chunk(x, 2, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        return feature, logic



class EntityIntersection(nn.Module):
    def __init__(self, dim):
        super(EntityIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, axis_embeddings, arg_embeddings):
        # N x B x d
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings


class EntityUnion(nn.Module):
    def __init__(self, dim):
        super(EntityUnion, self).__init__()
        self.dim = dim

    def forward(self, axis_embeddings, arg_embeddings):
        return axis_embeddings, arg_embeddings


class EntityNegation(nn.Module):
    def __init__(self, dim):
        super(EntityNegation, self).__init__()
        self.dim = dim

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding


class TimeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(TimeProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self,
                q1_feature, q1_logic,
                r_feature, r_logic,
                q2_feature, q2_logic):
        raise NotImplementedError("static query embeddings are unable to handle temporal information")


class TemporalIntersection(nn.Module):
    def __init__(self, dim):
        super(TemporalIntersection, self).__init__()
        self.dim = dim

    def forward(self, feature, logic):
        raise NotImplementedError("static query embeddings are unable to handle temporal information")


class TemporalUnion(nn.Module):
    def __init__(self, dim):
        super(TemporalUnion, self).__init__()
        self.dim = dim

    def forward(self, feature, logic):
        raise NotImplementedError("static query embeddings are unable to handle temporal information")


class TemporalNegation(nn.Module):
    def __init__(self, dim):
        super(TemporalNegation, self).__init__()
        self.dim = dim

    def forward(self, feature, logic):
        return feature, logic


class TemporalBefore(nn.Module):
    def __init__(self, dim):
        super(TemporalBefore, self).__init__()
        self.dim = dim

    def forward(self, feature, logic):
        return feature, logic


class TemporalAfter(nn.Module):
    def __init__(self, dim):
        super(TemporalAfter, self).__init__()
        self.dim = dim

    def forward(self, feature, logic):
        return feature, logic


class TemporalNext(nn.Module):
    def __init__(self):
        super(TemporalNext, self).__init__()

    def forward(self, feature, logic):
        return feature, logic



class TCQE(nn.Module):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(TCQE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.timestamp_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)

        self.timestamp_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)

        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation(hidden_dim)

        self.time_projection = TimeProjection(hidden_dim, drop=drop)
        self.time_intersection = TemporalIntersection(hidden_dim)
        self.time_union = TemporalUnion(hidden_dim)
        self.time_negation = TemporalNegation(hidden_dim)
        self.time_before = TemporalBefore(hidden_dim)
        self.time_after = TemporalAfter(hidden_dim)
        self.time_next = TemporalNext()

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
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            return self.entity_intersection(feature, logic)

        def And3(q1, q2, q3):
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            q3_feature, q3_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            return self.entity_intersection(feature, logic)

        def Or(q1, q2):
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            return self.entity_union(feature, logic)

        def Not(q):
            feature, logic = q
            return self.entity_negation(feature, logic)

        def EntityProjection2(e1, r1, t1):
            s_feature, s_logic, s_time_feature = e1
            r_feature, r_logic, r_time_feature = r1
            t_feature, t_logic, t_time_feature = t1
            return self.entity_projection(
                s_feature, s_logic, s_time_feature,
                r_feature, r_logic, r_time_feature,
                t_feature, t_logic, t_time_feature
            )
        query_embedding_operator_dict = {
            "And": And,
            "And3": And3,
            "Or": Or,
            "Not": Not,
            "EntityProjection": EntityProjection2,
        }
        return self.build_static_query_embedding_parser(query_embedding_operator_dict)

    def build_static_query_embedding_parser(self, query_embedding_operator_dict: Dict[str, Callable]):
        def TimeNot(q):
            feature, logic = q
            return self.time_negation(feature, logic)

        def TimeProjection2(e1, r1, e2):
            s_feature, s_logic, s_time_feature = e1
            r_feature, r_logic, r_time_feature = r1
            o_feature, o_logic, o_time_feature = e2
            return self.time_projection(
                s_feature, s_logic, s_time_feature,
                r_feature, r_logic, r_time_feature,
                o_feature, o_logic, o_time_feature
            )

        def TimeAnd(q1, q2):
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            return self.time_intersection(feature, logic)

        def TimeAnd3(q1, q2, q3):
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            q3_feature, q3_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            return self.time_intersection(feature, logic)

        def TimeOr(q1, q2):
            q1_feature, q1_logic = q1
            q2_feature, q2_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            return self.time_union(feature, logic)

        def TimeBefore(q):
            feature, logic = q
            return self.time_before(feature, logic)

        def TimeAfter(q):
            feature, logic = q
            return self.time_after(feature, logic)

        def TimeNext(q):
            feature, logic = q
            return self.time_next(feature, logic)

        def beforePt(e1, r1, e2):
            return TimeBefore(TimeProjection2(e1, r1, e2))

        def afterPt(e1, r1, e2):
            return TimeAfter(TimeProjection2(e1, r1, e2))

        neural_ops = {
            "TimeProjection": TimeProjection2,
            "TimeAnd": TimeAnd,
            "TimeAnd3": TimeAnd3,
            "TimeOr": TimeOr,
            "TimeNot": TimeNot,
            "TimeBefore": TimeBefore,
            "TimeAfter": TimeAfter,
            "TimeNext": TimeNext,
            "afterPt": afterPt,
            "beforePt": beforePt,
        }
        return expression.NeuralParser(dict(**neural_ops, **query_embedding_operator_dict))

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.timestamp_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range * 1.0 # it's 1.0 in ConE, not pi

    def entity_feature(self, idx):
        return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))

    def timestamp_feature(self, idx):
        return convert_to_time_feature(self.scale(self.timestamp_feature_embedding(idx)))

    def entity_token(self, idx) -> TYPE_token:
        feature = self.entity_feature(idx)
        logic = torch.zeros_like(feature).to(feature.device)
        return feature, logic

    def relation_token(self, idx) -> TYPE_token:
        feature = convert_to_feature(self.scale(self.relation_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.relation_logic_embedding(idx)))
        return feature, logic

    def timestamp_token(self, idx) -> TYPE_token:
        time_feature = self.timestamp_feature(idx)
        feature = torch.zeros_like(time_feature).to(time_feature.device)
        logic = torch.zeros_like(feature).to(feature.device)
        return feature, logic

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

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_FLEX(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

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

        return positive_scores, negative_scores, subsampling_weight, all_idxs

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
            for x in token_list:
                feature.append(x[0])
                logic.append(x[1])
            feature = torch.cat(feature, dim=0).unsqueeze(1)
            logic = torch.cat(logic, dim=0).unsqueeze(1)
            return feature, logic

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
            all_union_predict: TYPE_token = tuple([torch.stack([x, y], dim=1) for x, y in zip(predict_1, predict_2)])  # (B, 2, d) * 5
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(answer, all_union_predict, predict_entity=True, DNF_predict=True)
            else:
                return self.scoring_to_answers(answer, all_union_predict, predict_entity=False, DNF_predict=True)
        else:
            # other query and DM are normal
            func = self.parser.fast_function(query_name)
            embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
            predict = func(*embedding_of_args)  # (B, d)
            all_predict: TYPE_token = tuple([i.unsqueeze(dim=1) for i in predict])  # (B, 1, d)
            if is_to_predict_entity_set(query_name):
                return self.scoring_to_answers(answer, all_predict, predict_entity=True, DNF_predict=False)
            else:
                return self.scoring_to_answers(answer, all_predict, predict_entity=False, DNF_predict=False)

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
        all_predict:  (B, 1, dt) or (B, 2, dt) float
        return score: (B, N) float
        """
        q: TYPE_token = tuple([i.unsqueeze(dim=2) for i in q])  # (B, 1, 1, dt) or (B, 2, 1, dt)
        if predict_entity:
            feature = self.entity_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
            scores = self.scoring_entity(feature, q)  # (B, 1, N) or (B, 2, N)
        else:
            feature = self.timestamp_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
            scores = self.scoring_timestamp(feature, q)  # (B, 1, N) or (B, 2, N)

        if DNF_predict:
            scores = torch.max(scores, dim=1)[0]  # (B, N)
        else:
            scores = scores.squeeze(dim=1)  # (B, N)
        return scores  # (B, N)

    def distance_between_entity_and_query(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        """
        entity_embedding     (B, 1, N, d)
        query_axis_embedding (B, 1, 1, dt) or (B, 2, 1, dt)
        query_arg_embedding  (B, 1, 1, dt) or (B, 2, 1, dt)
        """
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        return distance

    def distance_between_timestamp_and_query(self, timestamp_feature):
        """
        timestamp_feature (B, 1, N, d)
        """
        d_center = timestamp_feature

        # inner distance
        feature_distance = torch.abs(d_center)
        distance = torch.norm(feature_distance, p=1, dim=-1)
        return distance

    def scoring_entity(self, entity_feature, q: TYPE_token):
        feature, logic = q
        distance = self.distance_between_entity_and_query(entity_feature, feature, logic)
        score = self.gamma - distance * self.modulus
        return score

    def scoring_timestamp(self, timestamp_feature, q: TYPE_token):
        feature, logic = q
        distance = self.distance_between_timestamp_and_query(timestamp_feature)
        score = self.gamma - distance * self.modulus
        return score


class TFLEX(TCQE):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(TFLEX, self).__init__(nentity, nrelation, ntimestamp, hidden_dim, gamma, test_batch_size, center_reg, drop)


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15, GDELT.")
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
@click.option('--train_tasks', type=str, default=
              "Pe,Pe2,Pe3,e2i,e3i,e2i_Pe,Pe_e2i,"
              + "e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe", help='the tasks for training')
@click.option('--train_all', type=bool, default=False, help='if training all, it will use all tasks in data.train_queries_answers')
@click.option('--eval_tasks', type=str, default="Pe,Pt,Pe2,Pe3", help='the tasks for evaluation')
@click.option('--eval_all', type=bool, default=False, help='if evaluating all, it will use all tasks in data.test_queries_answers')
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
         ):
    set_seeds(0)
    output = OutputSchema(dataset + "-" + name)

    if dataset == "ICEWS14":
        dataset = ICEWS14(data_home)
    elif dataset == "ICEWS05_15":
        dataset = ICEWS05_15(data_home)
    elif dataset == "GDELT":
        dataset = GDELT(data_home)
    cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
    data = ComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
        "train_queries_answers", "valid_queries_answers", "test_queries_answers",
    ])

    entity_count = data.entity_count
    relation_count = data.relation_count
    timestamp_count = data.timestamp_count
    max_relation_id = relation_count
    model = TFLEX(
        nentity=entity_count,
        nrelation=relation_count + max_relation_id,  # with reverse relations
        ntimestamp=timestamp_count,
        hidden_dim=hidden_dim,
        gamma=gamma,
        center_reg=center_reg,
        test_batch_size=test_batch_size,
        drop=input_dropout,
    )
    MyExperiment(
        output, data, model,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
    )


if __name__ == '__main__':
    main()
