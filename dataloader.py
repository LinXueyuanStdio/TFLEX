#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ComplexQueryData import QueryStructure
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
        negative_sample = torch.from_numpy(negative_sample)  # (self.negative_sample_size,)
        positive_sample = torch.LongTensor([tail])  # (1,)
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # positive_sample    : torch.LongTensor (1,)
        # negative_sample    : torch.LongTensor (self.negative_sample_size,)
        # subsampling_weight : torch.FloatTensor (1,)
        # flatten(query)     : List[**int]
        # query_structure    : List[**str]
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


class QueryStructureScoringAllDataset(Dataset):
    def __init__(self, queries: List[List[int]], nentity, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.entity_count = nentity
        self.answer = answer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query: List[int] = self.queries[idx]  # List[**int]
        data = torch.zeros(self.entity_count).float()
        data[list(self.answer[query])] = 1.
        # flatten ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')) -> ["e", "r", "n", "e", "r", "n", "n", "r"]
        # data    : torch.LongTensor (N,) where positive 1 while negative 0
        # flatten(query)     : List[**int]
        return data, flatten(query)

    @staticmethod
    def collate_fn(data):
        sample = torch.cat([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        return sample, query


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
