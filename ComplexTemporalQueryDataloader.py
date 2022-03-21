"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/16
@description: null
"""
import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from ComplexTemporalQueryData import TYPE_train_queries_answers, TYPE_test_queries_answers


def flatten_train(queries_answers: TYPE_train_queries_answers) -> List[Tuple[str, List[int], Set[int]]]:
    res = []
    for query_name, query_schema in queries_answers.items():
        qa_list: List[Tuple[List[int], Set[int]]] = query_schema["queries_answers"]
        for query, answer in qa_list:
            res.append((query_name, query, answer))
    return res


class TrainDataset(Dataset):
    def __init__(self, queries_answers: TYPE_train_queries_answers, nentity: int, negative_sample_size: int):
        self.all_data: List[Tuple[str, List[int], Set[int]]] = flatten_train(queries_answers)
        random.shuffle(self.all_data)
        self.len: int = len(self.all_data)
        self.nentity: int = nentity
        self.negative_sample_size: int = negative_sample_size
        self.count: Dict[str, int] = self.count_frequency(self.all_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, query, answer = deepcopy(self.all_data[idx])
        tail = np.random.choice(list(answer))  # select one answer
        subsampling_weight = self.count[query_name]  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_answer = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(negative_answer, answer, assume_unique=True, invert=True)
            negative_answer = negative_answer[mask]
            negative_sample_list.append(negative_answer)
            negative_sample_size += negative_answer.size
        negative_answer = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_answer = torch.from_numpy(negative_answer)  # (self.negative_sample_size,)
        positive_answer = torch.LongTensor([tail])  # (1,)
        # query = torch.LongTensor(query)  # (N,)
        #                                                                     (s, r, o, t) == (1, 7, 8, 5)
        # query_name         : str                                            'Pe'
        # args               : List[**str]                                    ['e1', 'r1', 't1']
        # query              : List[**int]                                    [ 1,    7,    5  ]
        # positive_answer    : torch.LongTensor (1,)                          Tensor([8])
        # negative_answer    : torch.LongTensor (self.negative_sample_size,)  Tensor([100, 392, 499, ...])
        # subsampling_weight : torch.FloatTensor (1,)                         Tensor([0.02])
        return query_name, query, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def collate_fn(data):
        positive_answer = torch.cat([_[2] for _ in data], dim=0)
        negative_answer = torch.stack([_[3] for _ in data], dim=0)
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        batch_queries_dict: Dict[str, List[List[int]]] = defaultdict(list)
        batch_idxs_dict: Dict[str, List[List[int]]] = defaultdict(list)
        for i, (query_name, query, _, _, _) in enumerate(data):
            batch_queries_dict[query_name].append(query)
            batch_idxs_dict[query_name].append([i])
        return batch_queries_dict, batch_idxs_dict, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def count_frequency(all_data: List[Tuple[str, List[int], Set[int]]], start=4) -> Dict[str, int]:
        count = {}
        for query_name, query, answer in all_data:
            count[query_name] = start + len(answer)
        return count


def flatten_test(queries_answers: TYPE_test_queries_answers) -> List[Tuple[str, List[int], Set[int], Set[int]]]:
    res = []
    for query_name, query_schema in queries_answers.items():
        qa_list: List[Tuple[List[int], Set[int], Set[int]]] = query_schema["queries_answers"]
        for query, easy_answer, hard_answer in qa_list:
            res.append((query_name, query, easy_answer, hard_answer))
    return res


class TestDataset(Dataset):
    def __init__(self, queries_answers: TYPE_test_queries_answers, nentity):
        self.all_data: List[Tuple[str, List[int], Set[int], Set[int]]] = flatten_test(queries_answers)
        random.shuffle(self.all_data)
        self.len: int = len(self.all_data)
        self.nentity: int = nentity

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, query, easy_answer, hard_answer = self.all_data[idx]
        # query = torch.LongTensor(query)  # (N,)
        if len(easy_answer) >= len(hard_answer):
            easy_answer = set()
        hard_answer = set(hard_answer) - set(easy_answer)
        candidate_answer = torch.LongTensor(range(self.nentity))
        return query_name, query, candidate_answer, easy_answer, hard_answer

    @staticmethod
    def collate_fn(data):
        query_name_list = [_[0] for _ in data]
        candidate_answer = torch.stack([_[2] for _ in data], dim=0)
        easy_answer = [_[3] for _ in data]
        hard_answer = [_[4] for _ in data]
        batch_queries_dict: Dict[str, List[List[int]]] = defaultdict(list)
        batch_idxs_dict: Dict[str, List[int]] = defaultdict(list)
        for i, (query_name, query, _, _, _) in enumerate(data):
            batch_queries_dict[query_name].append(query)
            batch_idxs_dict[query_name].append(i)
        return query_name_list, batch_queries_dict, batch_idxs_dict, candidate_answer, easy_answer, hard_answer
