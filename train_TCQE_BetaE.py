"""
@date: 2021/10/26
@description: null
"""
from typing import Tuple

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from ComplexTemporalQueryData import ICEWS05_15, ICEWS14, ComplexTemporalQueryDatasetCachePath, TemporalComplexQueryData, GDELT
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.nn.BetaE import BetaIntersection, BetaProjection, Regularizer
from toolbox.utils.RandomSeeds import set_seeds
from train_TCQE_TFLEX import MyExperiment
from TCQE_static_QE import TYPE_token, TCQE


class EntityProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(EntityProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 2
        entity_dim = token_dim
        relation_dim = token_dim

        self.projection_regularizer = Regularizer(1, 0.05, 1e9)  # make sure the parameters of beta embeddings after relation projection are positive
        self.projection_net = BetaProjection(entity_dim,
                                             relation_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)

    def forward(self,
                q_alpha, q_beta,
                r_alpha, r_beta,
                t_alpha, t_beta):
        q_embedding = torch.cat([q_alpha, q_beta], dim=-1)
        r_embedding = torch.cat([r_alpha, r_beta], dim=-1)

        x = self.projection_net(q_embedding, r_embedding)

        alpha_embedding, beta_embedding = torch.chunk(x, 2, dim=-1)

        return alpha_embedding, beta_embedding


class EntityIntersection(nn.Module):
    def __init__(self, dim):
        super(EntityIntersection, self).__init__()
        self.dim = dim
        self.center_net = BetaIntersection(dim)

    def forward(self, alpha_embeddings, beta_embeddings):
        # N x B x d
        alpha_embeddings, beta_embeddings = self.center_net(alpha_embeddings, beta_embeddings)
        return alpha_embeddings, beta_embeddings


class EntityUnion(nn.Module):
    def __init__(self, dim):
        super(EntityUnion, self).__init__()
        self.dim = dim

    def forward(self, alpha_embeddings, beta_embeddings):
        return alpha_embeddings, beta_embeddings


class EntityNegation(nn.Module):
    def __init__(self, dim):
        super(EntityNegation, self).__init__()
        self.dim = dim

    def forward(self, alpha_embedding, beta_embedding):
        alpha_embedding = 1. / alpha_embedding
        beta_embedding = 1. / beta_embedding

        return alpha_embedding, beta_embedding


class TFLEX(TCQE):
    def __init__(self, nentity, nrelation, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1,
                 center_reg=None, drop: float = 0.):
        super(TFLEX, self).__init__(nentity, nrelation, ntimestamp, hidden_dim, gamma, test_batch_size, center_reg, drop)
        self.entity_feature_embedding = nn.Embedding(nentity, hidden_dim * 2) # [alpha; beta]
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)  # make sure the parameters of beta embeddings are positive

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation(hidden_dim)

    def entity_feature(self, idx):
        return self.entity_regularizer(self.entity_feature_embedding(idx))

    def entity_token(self, idx) -> TYPE_token:
        entity_embedding = self.entity_feature(idx)
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        return alpha_embedding, beta_embedding

    def relation_token(self, idx) -> TYPE_token:
        alpha_embedding = self.relation_feature_embedding(idx)
        beta_embedding = self.relation_logic_embedding(idx)
        return alpha_embedding, beta_embedding

    def distance_between_entity_and_query(self, entity_embedding, query_alpha_embedding, query_beta_embedding):
        """
        entity_embedding     (B, 1, N, d)
        query_axis_embedding (B, 1, 1, dt) or (B, 2, 1, dt)
        query_arg_embedding  (B, 1, 1, dt) or (B, 2, 1, dt)
        """
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        query_dist = torch.distributions.beta.Beta(query_alpha_embedding, query_beta_embedding)
        distance = torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
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
@click.option('--cpu_num', type=int, default=1, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=15.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--train_tasks', type=str, default=
              "Pe,Pe2,Pe3,e2i,e3i,"
              + "e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe", help='the tasks for training')
@click.option('--train_all', type=bool, default=False, help='if training all, it will use all tasks in data.train_queries_answers')
@click.option('--eval_tasks', type=str, default="Pe,Pe2,Pe3", help='the tasks for evaluation')
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
    data = TemporalComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache(["meta"])

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
