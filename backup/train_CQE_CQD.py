"""
@date: 2021/10/26
@description: null
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Callable, Optional

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from backup import discrete as d2
from ComplexQueryData import *
from backup.dataloader import TestDataset
from toolbox.data.dataloader import SingledirectionalOneShotIterator
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds
from backup.util import flatten_query, flatten


class CQDTrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.answer = answer

        self.qa_lst = []
        for q, qs in queries:
            for a in self.answer[q]:
                qa_entry = (qs, q, a)
                self.qa_lst += [qa_entry]

        self.qa_len = len(self.qa_lst)

    def __len__(self):
        # return self.len
        return self.qa_len

    def __getitem__(self, idx):
        # query = self.queries[idx][0]
        query = self.qa_lst[idx][1]

        # query_structure = self.queries[idx][1]
        query_structure = self.qa_lst[idx][0]

        # tail = np.random.choice(list(self.answer[query]))
        tail = self.qa_lst[idx][2]

        # subsampling_weight = self.count[query]
        # subsampling_weight = torch.sqrt(1 / Tensor([subsampling_weight]))
        subsampling_weight = torch.tensor([1.0])

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure


def flatten_structure(query_structure):
    if type(query_structure) == str:
        return [query_structure]

    flat_structure = []
    for element in query_structure:
        flat_structure.extend(flatten_structure(element))

    return flat_structure


def query_to_atoms(query_structure, flat_ids):
    flat_structure = flatten_structure(query_structure)
    batch_size, query_length = flat_ids.shape
    assert len(flat_structure) == query_length

    query_triples = []
    variable = 0
    previous = flat_ids[:, 0]
    conjunction_mask = []
    negation_mask = []

    for i in range(1, query_length):
        if flat_structure[i] == 'r':
            variable -= 1
            triples = torch.empty(batch_size, 3,
                                  device=flat_ids.device,
                                  dtype=torch.long)
            triples[:, 0] = previous
            triples[:, 1] = flat_ids[:, i]
            triples[:, 2] = variable

            query_triples.append(triples)
            previous = variable
            conjunction_mask.append(True)
            negation_mask.append(False)
        elif flat_structure[i] == 'e':
            previous = flat_ids[:, i]
            variable += 1
        elif flat_structure[i] == 'u':
            conjunction_mask = [False] * len(conjunction_mask)
        elif flat_structure[i] == 'n':
            negation_mask[-1] = True

    atoms = torch.stack(query_triples, dim=1)
    num_variables = variable * -1
    conjunction_mask = torch.tensor(conjunction_mask).unsqueeze(0).expand(batch_size, -1)
    negation_mask = torch.tensor(negation_mask).unsqueeze(0).expand(batch_size, -1)

    return atoms, num_variables, conjunction_mask, negation_mask


def create_instructions(chains):
    instructions = []

    prev_start = None
    prev_end = None

    path_stack = []
    start_flag = True
    for chain_ind, chain in enumerate(chains):
        if start_flag:
            prev_end = chain[-1]
            start_flag = False
            continue

        if prev_end == chain[0]:
            instructions.append(f"hop_{chain_ind - 1}_{chain_ind}")
            prev_end = chain[-1]
            prev_start = chain[0]

        elif prev_end == chain[-1]:

            prev_start = chain[0]
            prev_end = chain[-1]

            instructions.append(f"intersect_{chain_ind - 1}_{chain_ind}")
        else:
            path_stack.append(([prev_start, prev_end], chain_ind - 1))
            prev_start = chain[0]
            prev_end = chain[-1]
            start_flag = False
            continue

        if len(path_stack) > 0:

            path_prev_start = path_stack[-1][0][0]
            path_prev_end = path_stack[-1][0][-1]

            if path_prev_end == chain[-1]:
                prev_start = chain[0]
                prev_end = chain[-1]

                instructions.append(f"intersect_{path_stack[-1][1]}_{chain_ind}")
                path_stack.pop()
                continue

    ans = []
    for inst in instructions:
        if ans:

            if 'inter' in inst and ('inter' in ans[-1]):
                last_ind = inst.split("_")[-1]
                ans[-1] = ans[-1] + f"_{last_ind}"
            else:
                ans.append(inst)

        else:
            ans.append(inst)

    instructions = ans
    return instructions


def t_norm_fn(tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
    if 'min' in t_norm:
        return torch.min(tens_1, tens_2)
    elif 'prod' in t_norm:
        return tens_1 * tens_2


def t_conorm_fn(tens_1: Tensor, tens_2: Tensor, t_norm: str = 'min') -> Tensor:
    if 'min' in t_norm:
        return torch.max(tens_1, tens_2)
    elif 'prod' in t_norm:
        return (tens_1 + tens_2) - (tens_1 * tens_2)


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    max_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, max_batch)]
    return res


def get_best_candidates(rel: Tensor,
                        arg1: Tensor,
                        forward_emb: Callable[[Tensor, Tensor], Tensor],
                        entity_embeddings: Callable[[Tensor], Tensor],
                        candidates: int = 5,
                        last_step: bool = False) -> Tuple[Tensor, Tensor]:
    batch_size, embedding_size = rel.shape[0], rel.shape[1]

    # [B, N]
    scores = forward_emb(arg1, rel)

    if not last_step:
        # [B, K], [B, K]
        k = min(candidates, scores.shape[1])
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)
        # [B, K, E]
        z_emb = entity_embeddings(z_indices)

        # XXX: move before return
        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size
    else:
        z_scores = scores

        z_indices = torch.arange(z_scores.shape[1]).view(1, -1).repeat(z_scores.shape[0], 1).to(rel.device)
        z_emb = entity_embeddings(z_indices)

    return z_scores, z_emb


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQD(nn.Module):
    MIN_NORM = 'min'
    PROD_NORM = 'prod'
    NORMS = {MIN_NORM, PROD_NORM}

    def __init__(self,
                 nentity: int,
                 nrelation: int,
                 rank: int,
                 init_size: float = 1e-3,
                 reg_weight: float = 1e-2,
                 method: str = 'discrete',
                 t_norm_name: str = 'prod',
                 k: int = 5,
                 query_name_dict: Optional[Dict] = None,
                 do_sigmoid: bool = False,
                 do_normalize: bool = False,
                 ):
        super(CQD, self).__init__()

        self.rank = rank
        self.nentity = nentity
        self.nrelation = nrelation
        self.method = method
        self.t_norm_name = t_norm_name
        self.k = k
        self.query_name_dict = query_name_dict

        sizes = (nentity, nrelation)
        self.init_size = init_size
        # self.entity_embedding_a = nn.Embedding(nentity, self.rank)
        # self.entity_embedding_b = nn.Embedding(nentity, self.rank)
        # self.relation_embedding_a = nn.Embedding(nrelation, self.rank)
        # self.relation_embedding_b = nn.Embedding(nrelation, self.rank)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 2 * rank, sparse=False) for s in sizes[:2]])

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        self.do_sigmoid = do_sigmoid
        self.do_normalize = do_normalize

    def init(self):
        # self.entity_embedding_a.weight.data *= self.init_size
        # self.entity_embedding_b.weight.data *= self.init_size
        # self.relation_embedding_a.weight.data *= self.init_size
        # self.relation_embedding_b.weight.data *= self.init_size
        self.embeddings[0].weight.data *= self.init_size
        self.embeddings[1].weight.data *= self.init_size

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        lhs = lhs_emb[..., :self.rank], lhs_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]
        rhs = rhs_emb[..., :self.rank], rhs_emb[..., self.rank:]
        return lhs, rel, rhs

    def loss(self, triples: Tensor) -> Tensor:
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)
        return l_fit + l_reg

    def embed_entity(self, idx: Tensor):
        return self.entity_embedding_a(idx), self.entity_embedding_b(idx)

    def embed_relation(self, idx: Tensor):
        return self.relation_embedding_a(idx), self.relation_embedding_b(idx)

    def score_candidates(self, triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Optional[List[Tensor]]]:
        lhs_emb = self.embeddings[0](triples[:, 0])
        rel_emb = self.embeddings[1](triples[:, 1])
        rhs_emb = self.embeddings[0](triples[:, 2])
        to_score = self.embeddings[0].weight
        scores_o, _ = self.score_o(lhs_emb, rel_emb, to_score)
        scores_s, _ = self.score_s(to_score, rel_emb, rhs_emb)
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        factors = self.get_factors(lhs, rel, rhs)
        return (scores_o, scores_s), factors

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def score_s(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def get_full_embeddings(self, queries: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        lhs = rel = rhs = None
        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        if torch.sum(queries[:, 1]).item() > 0:
            rel = self.embeddings[1](queries[:, 1])
        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        return lhs, rel, rhs

    def batch_t_norm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.min(scores, dim=1)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.prod(scores, dim=1)
        else:
            raise ValueError(f't_norm must be one of {CQD.NORMS}, got {self.t_norm_name}')

        return scores

    def batch_t_conorm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.max(scores, dim=1, keepdim=True)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.sum(scores, dim=1, keepdim=True) - torch.prod(scores, dim=1, keepdim=True)
        else:
            raise ValueError(f't_norm must be one of {CQD.NORMS}, got {self.t_norm_name}')

        return scores

    def reduce_query_score(self, atom_scores, conjunction_mask, negation_mask):
        batch_size, num_atoms, *extra_dims = atom_scores.shape

        atom_scores = torch.sigmoid(atom_scores)
        scores = atom_scores.clone()
        scores[negation_mask] = 1 - atom_scores[negation_mask]

        disjunctions = scores[~conjunction_mask].reshape(batch_size, -1, *extra_dims)
        conjunctions = scores[conjunction_mask].reshape(batch_size, -1, *extra_dims)

        if disjunctions.shape[1] > 0:
            disjunctions = self.batch_t_conorm(disjunctions)

        conjunctions = torch.cat([disjunctions, conjunctions], dim=1)

        t_norm = self.batch_t_norm(conjunctions)
        return t_norm

    def forward(self,
                positive_sample,
                negative_sample,
                subsampling_weight,
                batch_queries_dict: Dict[Tuple, Tensor],
                batch_idxs_dict):
        all_idxs = []
        all_scores = []

        scores = None

        for query_structure, queries in batch_queries_dict.items():
            batch_size = queries.shape[0]
            atoms, num_variables, conjunction_mask, negation_mask = query_to_atoms(query_structure, queries)

            all_idxs.extend(batch_idxs_dict[query_structure])

            # [False, True]
            target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

            # Offsets identify variables across different batches
            var_id_offsets = torch.arange(batch_size, device=atoms.device) * num_variables
            var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

            # Replace negative variable IDs with valid identifiers
            vars_mask = atoms < 0
            atoms_offset_vars = -atoms + var_id_offsets

            atoms[vars_mask] = atoms_offset_vars[vars_mask]

            head, rel, tail = atoms[..., 0], atoms[..., 1], atoms[..., 2]
            head_vars_mask = vars_mask[..., 0]

            with torch.no_grad():
                h_emb_constants = self.embeddings[0](head)
                r_emb = self.embeddings[1](rel)

            if 'continuous' in self.method:
                h_emb = h_emb_constants
                if num_variables > 1:
                    # var embedding for ID 0 is unused for ease of implementation
                    var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
                    var_embs.weight.data *= self.init_size

                    var_embs.to(atoms.device)
                    optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
                    prev_loss_value = 1000
                    loss_value = 999
                    i = 0

                    # CQD-CO optimization loop
                    while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        h_emb = h_emb_constants.clone()
                        # Fill variable positions with optimizable embeddings
                        h_emb[head_vars_mask] = var_embs(head[head_vars_mask])

                        t_emb = var_embs(tail)
                        scores, factors = self.score_o(h_emb.unsqueeze(-2),
                                                       r_emb.unsqueeze(-2),
                                                       t_emb.unsqueeze(-2),
                                                       return_factors=True)

                        query_score = self.reduce_query_score(scores,
                                                              conjunction_mask,
                                                              negation_mask)

                        loss = - query_score.mean() + self.regularizer.forward(factors)
                        loss_value = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        i += 1

                with torch.no_grad():
                    # Select predicates involving target variable only
                    conjunction_mask = conjunction_mask[target_mask].reshape(batch_size, -1)
                    negation_mask = negation_mask[target_mask].reshape(batch_size, -1)

                    target_mask = target_mask.unsqueeze(-1).expand_as(h_emb)
                    emb_size = h_emb.shape[-1]
                    h_emb = h_emb[target_mask].reshape(batch_size, -1, emb_size)
                    r_emb = r_emb[target_mask].reshape(batch_size, -1, emb_size)
                    to_score = self.embeddings[0].weight

                    scores, factors = self.score_o(h_emb, r_emb, to_score)
                    query_score = self.reduce_query_score(scores,
                                                          conjunction_mask,
                                                          negation_mask)
                    all_scores.append(query_score)

                scores = torch.cat(all_scores, dim=0)

            elif 'discrete' in self.method:
                graph_type = query_name_dict[query_structure]

                def t_norm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.minimum(a, b)

                def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.maximum(a, b)

                if self.t_norm_name == CQD.PROD_NORM:
                    def t_norm(a: Tensor, b: Tensor) -> Tensor:
                        return a * b

                    def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                        return 1 - ((1 - a) * (1 - b))

                def normalize(scores_: Tensor) -> Tensor:
                    scores_ = scores_ - scores_.min(1, keepdim=True)[0]
                    scores_ = scores_ / scores_.max(1, keepdim=True)[0]
                    return scores_

                def scoring_function(rel_: Tensor, lhs_: Tensor, rhs_: Tensor) -> Tensor:
                    res, _ = self.score_o(lhs_, rel_, rhs_)
                    if self.do_sigmoid is True:
                        res = torch.sigmoid(res)
                    if self.do_normalize is True:
                        res = normalize(res)
                    return res

                if graph_type == "1p":
                    scores = d2.query_1p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function)
                elif graph_type == "2p":
                    scores = d2.query_2p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "3p":
                    scores = d2.query_3p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2i":
                    scores = d2.query_2i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "3i":
                    scores = d2.query_3i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "pi":
                    scores = d2.query_pi(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "ip":
                    scores = d2.query_ip(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2u-DNF":
                    scores = d2.query_2u_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             t_conorm=t_conorm)
                elif graph_type == "up-DNF":
                    scores = d2.query_up_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             k=self.k, t_norm=t_norm, t_conorm=t_conorm)
                else:
                    raise ValueError(f'Unknown query type: {graph_type}')

        return None, scores, None, all_idxs


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * -1
    return y


class Projection(nn.Module):
    def __init__(self, dim, hidden_dim=1600, num_layers=2):
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
        # logic:  N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logic = torch.prod(logic, dim=0)
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
        self.entity_feature_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=True)
        self.relation_feature_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        self.relation_logic_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        self.projection = Projection(self.entity_dim)
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
        nn.init.uniform_(tensor=self.entity_feature_embedding, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_feature_embedding, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    # implement formatting forward method
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
            all_feature = torch.cat(all_feature, dim=0).unsqueeze(1)  # (B, 1, d)
            all_logic = torch.cat(all_logic, dim=0).unsqueeze(1)  # (B, 1, d)
        if len(all_union_feature) > 0:
            all_union_feature = torch.cat(all_union_feature, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_logic = torch.cat(all_union_logic, dim=0).unsqueeze(1)  # (2B, 1, d)
            all_union_feature = all_union_feature.view(all_union_feature.shape[0] // 2, 2, 1, -1)  # (B, 2, 1, d)
            all_union_logic = all_union_logic.view(all_union_logic.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # 2. 计算正例损失
        if type(positive_sample) != type(None):
            # 2.1 计算 一般的查询
            if len(all_feature) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]

                positive_feature = torch.index_select(self.entity_feature_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_feature = self.scale(positive_feature)
                positive_feature = convert_to_feature(positive_feature)

                positive_logit = self.cal_logit(positive_feature, all_feature, all_logic)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)

            # 2.1 计算 并查询
            if len(all_union_feature) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]

                positive_feature = torch.index_select(self.entity_feature_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_feature = self.scale(positive_feature)
                positive_feature = convert_to_feature(positive_feature)

                positive_union_logit = self.cal_logit(positive_feature, all_union_feature, all_union_logic)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        # 3. 计算负例损失
        if type(negative_sample) != type(None):
            # 3.1 计算 一般的查询
            if len(all_feature) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape

                negative_feature = torch.index_select(self.entity_feature_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_feature = self.scale(negative_feature)
                negative_feature = convert_to_feature(negative_feature)

                negative_logit = self.cal_logit(negative_feature, all_feature, all_logic)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)

            # 3.1 计算 并查询
            if len(all_union_feature) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape

                negative_feature = torch.index_select(self.entity_feature_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_feature = self.scale(negative_feature)
                negative_feature = convert_to_feature(negative_feature)

                negative_union_logit = self.cal_logit(negative_feature, all_union_feature, all_union_logic)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_feature_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def embed_query(self, queries: torch.Tensor, query_structure, idx: int):
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
                feature_entity_embedding = torch.index_select(self.entity_feature_embedding, dim=0, index=queries[:, idx])
                feature_entity_embedding = self.scale(feature_entity_embedding)
                feature_entity_embedding = convert_to_feature(feature_entity_embedding)

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
                    r_feature = torch.index_select(self.relation_feature_embedding, dim=0, index=queries[:, idx])
                    r_feature = self.scale(r_feature)
                    r_feature = convert_to_feature(r_feature)

                    r_logic = torch.index_select(self.relation_logic_embedding, dim=0, index=queries[:, idx])
                    r_logic = self.scale(r_logic)
                    r_logic = convert_to_feature(r_logic)

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

    # implement distance function
    def distance(self, entity_feature, query_feature, query_logic):
        # inner distance 这里 sin(x) 近似为 L1 范数
        distance2feature = torch.abs(entity_feature - query_feature)
        distance_base = torch.abs(query_logic)
        distance_in = torch.min(distance2feature, distance_base)

        # outer distance
        delta1 = entity_feature - (query_feature - query_logic)
        delta2 = entity_feature - (query_feature + query_logic)
        indicator_in = distance2feature < distance_base
        distance_out = torch.min(torch.abs(delta1), torch.abs(delta2))
        distance_out[indicator_in] = 0.

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        return distance

    def cal_logit(self, entity_feature, query_feature, query_logic):
        distance_1 = self.distance(entity_feature, query_feature, query_logic)
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
                 batch_size, valid_batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, tasks, evaluate_union, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg,
                 reg_weight, cqd_type, cqd_t_norm, cqd_k, cqd_sigmoid, cqd_normalize,
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

        remove_query_names = ["2u-DM", "up-DM"]
        for query_name in remove_query_names:
            query_structure = name_query_dict[query_name]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

        self.log("Training info:")
        for query_structure in train_queries:
            self.log(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        path_list = ['1p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            CQDTrainDataset(train_path_queries, nentity, nrelation, negative_sample_size, train_answers),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=CQDTrainDataset.collate_fn
        ))

        self.log("Validation info:")
        for query_structure in valid_queries:
            self.log(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(valid_queries, nentity, nrelation),
            batch_size=valid_batch_size,
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
        # model = FLEX(
        #     nentity=nentity,
        #     nrelation=nrelation,
        #     hidden_dim=hidden_dim,
        #     gamma=gamma,
        #     center_reg=center_reg,
        #     test_batch_size=test_batch_size,
        #     query_name_dict=query_name_dict,
        #     drop=input_dropout,
        # ).to(train_device)
        model = CQD(
            nentity=nentity,
            nrelation=nrelation,
            rank=hidden_dim,
            query_name_dict=query_name_dict,
            reg_weight=reg_weight,
            method=cqd_type,
            t_norm_name=cqd_t_norm,
            k=cqd_k,
            do_sigmoid=cqd_sigmoid,
            do_normalize=cqd_sigmoid,
        ).to(train_device)
        self.valid_batch_entity_range = torch.arange(nentity).float().repeat(valid_batch_size, 1).cuda()
        self.test_batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1).cuda()
        opt = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
                result = self.evaluate(model, valid_easy_answers, valid_hard_answers, valid_dataloader, valid_batch_size, test_device)
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

            progbar.update(step + 1, [("step", step + 1), ("loss", log["loss"]), ("positive", log["positive_sample_loss"]), ("negative", log["negative_sample_loss"])])
            if (step + 1) % 10 == 0:
                self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                print("")
                self.log('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                opt = torch.optim.Adagrad(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_easy_answers, valid_hard_answers, valid_dataloader, valid_batch_size, test_device)
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

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict: Dict[List[str], list] = defaultdict(list)
        batch_idxs_dict: Dict[List[str], List[int]] = defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device)
        positive_sample = positive_sample.to(device)

        input_batch = batch_queries_dict[('e', ('r',))]
        input_batch = torch.cat((input_batch, positive_sample.unsqueeze(1)), dim=1)
        loss = model.loss(input_batch)
        positive_sample_loss = negative_sample_loss = loss

        optimizer.zero_grad()
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
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(list)
        step = 0
        h10 = None
        batch_queries_dict = defaultdict(list)
        batch_idxs_dict = defaultdict(list)
        for negative_sample, queries, queries_unflatten, query_structures in test_dataloader:
            batch_queries_dict.clear()
            batch_idxs_dict.clear()
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
            ranking = argsort.float()
            if len(argsort) == test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                batch_entity_range = self.valid_batch_entity_range if test_batch_size == len(self.valid_batch_entity_range) else self.test_batch_entity_range
                ranking = ranking.scatter_(1, argsort, batch_entity_range)  # achieve the ranking of all entities
            else:  # otherwise, create a new torch Tensor for batch_entity_range
                scatter_src = torch.arange(model.nentity).float().repeat(argsort.shape[0], 1).cuda()
                # achieve the ranking of all entities
                ranking = ranking.scatter_(1, argsort, scatter_src)
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
@click.option("--data_home", type=str, default="data/reasoning", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, NELL.")
@click.option("--name", type=str, default="FLEX_base", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=100000, help="Number of steps.")
# @click.option("--warm_up_steps", type=int, default=100000000, help="Number of steps.")
@click.option("--every_test_step", type=int, default=1000, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=500, help="Number of steps.")
@click.option("--batch_size", type=int, default=2000, help="Batch size.")
@click.option("--valid_batch_size", type=int, default=1000, help="Test batch size.")
@click.option("--test_batch_size", type=int, default=1, help="Test batch size.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.1, help="Learning rate.")
@click.option('--tasks', type=str, default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
@click.option('--evaluate_union', type=str, default="DNF", help='choices=[DNF, DM] the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
@click.option('--cpu_num', type=int, default=4, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=1000, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=30.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--reg_weight', type=float, default=0.1)
@click.option('--cqd_type', type=click.Choice(['continuous', 'discrete']), default='discrete')
@click.option('--cqd_t_norm', type=click.Choice(list(CQD.NORMS)), default=CQD.PROD_NORM)
@click.option('--cqd_k', type=int, default=5)
@click.option('--cqd_sigmoid', is_flag=True)
@click.option('--cqd_normalize', is_flag=True)
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, valid_batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, tasks, evaluate_union, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg,
         reg_weight, cqd_type, cqd_t_norm, cqd_k, cqd_sigmoid, cqd_normalize,
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
        batch_size, valid_batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, tasks, evaluate_union, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg,
        reg_weight, cqd_type, cqd_t_norm, cqd_k, cqd_sigmoid, cqd_normalize,
    )


if __name__ == '__main__':
    main()
