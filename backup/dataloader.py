"""
@date: 2021/10/26
@description: null
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from typing import Tuple, List, Set, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from ComplexQueryData import QueryStructure, QueryFlattenIds, query_name_dict
from util import flatten


class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        return negative_sample, flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure


class TrainDataset(Dataset):
    def __init__(self, queries: List[Tuple[List[int], QueryStructure]], nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query: List[int] = self.queries[idx][0]  # List[**int]
        query_structure: QueryStructure = self.queries[idx][1]  # Tuple[**str]
        tail = np.random.choice(list(self.answer[query]))  # select one answer
        subsampling_weight = self.count[query]  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_answer = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_answer,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_answer = negative_answer[mask]
            negative_sample_list.append(negative_answer)
            negative_sample_size += negative_answer.size
        negative_answer = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_answer = torch.from_numpy(negative_answer)  # (self.negative_sample_size,)
        positive_answer = torch.LongTensor([tail])  # (1,)
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # positive_sample    : torch.LongTensor (1,)
        # negative_sample    : torch.LongTensor (self.negative_sample_size,)
        # subsampling_weight : torch.FloatTensor (1,)
        # flatten(query)     : List[**int]
        # query_structure    : List[**str]
        return positive_answer, negative_answer, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure


class DistributedTestDataset(Dataset):
    def __init__(self, queries: List[Tuple[QueryFlattenIds, QueryStructure]], nentity: int, nrelation: int, easy_answer_dict, hard_answer_dict):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries: List[Tuple[QueryFlattenIds, QueryStructure]] = queries
        self.nentity: int = nentity
        self.nrelation: int = nrelation
        self.easy_answer_dict = easy_answer_dict
        self.hard_answer_dict = hard_answer_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        easy_answer = self.easy_answer_dict[query]
        hard_answer = self.hard_answer_dict[query]
        query_structure = self.queries[idx][1]
        query_name = query_name_dict[query_structure]
        answer_range = self.nentity
        candidate_answer = torch.LongTensor(range(answer_range))
        query = flatten(query)
        easy_answer_vector = torch.zeros(answer_range)
        if len(easy_answer) > 0:
            easy_answer_vector[list(easy_answer)] = 1
        easy_answer_mask = easy_answer_vector == 1
        return query_name, query, candidate_answer, easy_answer_mask, hard_answer

    @staticmethod
    def collate_fn(data):
        grouped_query: Dict[str, List[List[int]]] = defaultdict(list)
        grouped_candidate_answer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        grouped_easy_answer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        grouped_hard_answer_dict: Dict[str, List[Set[int]]] = defaultdict(list)
        for i, (query_name, query, candidate_answer, easy_answer, hard_answer) in enumerate(data):
            if None in query:
                print("error", query_name, query)
            grouped_query[query_name].append(query)
            grouped_candidate_answer[query_name].append(candidate_answer)
            grouped_easy_answer_dict[query_name].append(easy_answer)
            grouped_hard_answer_dict[query_name].append(hard_answer)

        grouped_easy_answer: Dict[str, torch.Tensor] = {
            key: torch.stack(grouped_easy_answer_dict[key], dim=0)
            for key in grouped_easy_answer_dict
        }
        data_list = []
        for query_name in grouped_query:
            data_list.append((query_name, torch.LongTensor(grouped_query[query_name]), torch.stack(grouped_candidate_answer[query_name], dim=0)))
        return data_list, grouped_easy_answer, grouped_hard_answer_dict


class DistributedTrainDataset(Dataset):
    def __init__(self, queries: List[Tuple[QueryFlattenIds, QueryStructure]], nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query: QueryFlattenIds = self.queries[idx][0]  # List[**int]
        query_structure: QueryStructure = self.queries[idx][1]  # Tuple[**str]
        query_name = query_name_dict[query_structure]
        tail = np.random.choice(list(self.answer[query]))  # select one answer
        subsampling_weight = self.count[query]  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_answer = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_answer,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_answer = negative_answer[mask]
            negative_sample_list.append(negative_answer)
            negative_sample_size += negative_answer.size
        negative_answer = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_answer = torch.from_numpy(negative_answer)  # (self.negative_sample_size,)
        positive_answer = torch.LongTensor([tail])  # (1,)
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # positive_sample    : torch.LongTensor (1,)
        # negative_sample    : torch.LongTensor (self.negative_sample_size,)
        # subsampling_weight : torch.FloatTensor (1,)
        # flatten(query)     : List[**int]
        # query_structure    : List[**str]
        query_tensor = flatten(query)
        return query_name, query_tensor, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def collate_fn(data):
        positive_answer = torch.cat([_[2] for _ in data], dim=0)
        negative_answer = torch.stack([_[3] for _ in data], dim=0)
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        grouped_query: Dict[str, List[List[int]]] = defaultdict(list)
        grouped_idxs: Dict[str, List[int]] = defaultdict(list)
        for i, (query_name, query, _, _, _) in enumerate(data):
            grouped_query[query_name].append(query)
            grouped_idxs[query_name].append(i)
        data_list = []
        for query_name in grouped_query:
            idx = grouped_idxs[query_name]
            t = (query_name,
                 torch.LongTensor(grouped_query[query_name]),
                 positive_answer[idx].view(len(idx), -1),
                 negative_answer[idx].view(len(idx), -1),
                 subsampling_weight[idx])
            # [(query_name, query_tensor, positive_answer, negative_answer, subsampling_weight)]
            data_list.append(t)
        return data_list

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


class QueryStructureScoringAllDataset(Dataset):
    def __init__(self, queries: List[Tuple[QueryFlattenIds, Set[int]]], nentity):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.entity_count = nentity

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query, answer_ids = self.queries[idx]  # List[**int]
        answer = torch.zeros(self.entity_count).float()
        answer[list(answer_ids)] = 1.
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # flatten(query) : List[**int]
        # answer         : torch.LongTensor (N,) where positive 1 while negative 0
        flatten_query = torch.LongTensor(flatten(query))
        return flatten_query, answer


class QueryStructureScoringNDataset(Dataset):
    def __init__(self, queries: List[Tuple[QueryFlattenIds, Set[int]]], nentity, negative_sample_size):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.entity_count = nentity
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query, answer_ids = self.queries[idx]  # List[**int]

        positive_id = np.random.choice(list(answer_ids))  # select one answer
        subsampling_weight = len(answer_ids)  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.entity_count, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                answer_ids,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        selected_ids = [positive_id] + list(negative_sample)  # 1 vs. N
        sample = torch.LongTensor(selected_ids)

        answer = torch.zeros(len(selected_ids)).float()
        answer[0] = 1.
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # flatten(query) : List[**int]
        # answer         : torch.LongTensor (N,) where positive 1 while negative 0
        flatten_query = torch.LongTensor(flatten(query))
        return flatten_query, sample, answer, subsampling_weight


class QueryStructureScoringDataset(Dataset):
    def __init__(self, queries: List[Tuple[QueryFlattenIds, Set[int]]], nentity, negative_sample_size):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.entity_count = nentity
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query, answer_ids = self.queries[idx]  # List[**int]

        positive_id = np.random.choice(list(answer_ids))  # select one answer
        subsampling_weight = len(answer_ids)  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.entity_count, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                answer_ids,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        positive_sample = torch.LongTensor([positive_id])
        negative_sample = torch.LongTensor(negative_sample)
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # flatten(query) : List[**int]
        flatten_query = torch.LongTensor(flatten(query))
        return flatten_query, positive_sample, negative_sample, subsampling_weight


class ScoringAllDataset(Dataset):
    def __init__(self, queries: List[Tuple[List[int], QueryStructure]], nentity, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.entity_count = nentity
        self.answer = answer

        structures = set()
        self.structure2query = defaultdict(list)
        for q, s in queries:
            self.structure2query[s].append(q)
            structures.add(s)
        self.structures = list(structures)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        query_structure: QueryStructure = self.structures[idx]  # Tuple[**str]
        query: List[int] = self.queries[idx][0]  # List[**int]
        data = torch.zeros(self.entity_count).float()
        data[list(self.answer[query])] = 1.
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # data    : torch.LongTensor (N,) where positive 1 while negative 0
        # flatten(query)     : List[**int]
        # query_structure    : List[**str]
        return data, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        sample = torch.cat([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        return sample, query, query_structure


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
