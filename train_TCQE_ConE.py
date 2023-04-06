"""
@date: 2021/10/26
@description: null
"""
from typing import Tuple

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from ComplexTemporalQueryData import ICEWS05_15, ICEWS14, ComplexTemporalQueryDatasetCachePath, ComplexQueryData, GDELT
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.RandomSeeds import set_seeds
from train_TCQE_TFLEX import MyExperiment
from TCQE_static_QE import TYPE_token, TCQE


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


class TFLEX(TCQE):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(TFLEX, self).__init__(nentity, nrelation, ntimestamp, hidden_dim, gamma, test_batch_size, center_reg, drop)
        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation(hidden_dim)

    def entity_feature(self, idx):
        return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))

    def entity_token(self, idx) -> TYPE_token:
        feature = self.entity_feature(idx)
        logic = torch.zeros_like(feature).to(feature.device)
        return feature, logic

    def relation_token(self, idx) -> TYPE_token:
        feature = convert_to_feature(self.scale(self.relation_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.relation_logic_embedding(idx)))
        return feature, logic

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


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15, GDELT.")
@click.option("--name", type=str, default="TFLEX_base", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=200001, help="Number of steps.")
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
@click.option('--gamma', type=float, default=15.0, help="margin in the loss")
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
