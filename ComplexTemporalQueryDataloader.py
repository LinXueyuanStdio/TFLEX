"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/16
@description: null
"""
import random
from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from ComplexTemporalQueryData import TYPE_train_queries_answers, TYPE_test_queries_answers


def flatten_train(queries_answers: TYPE_train_queries_answers) -> List[Tuple[str, List[str], List[int], Set[int]]]:
    res = []
    for query_name, query_schema in queries_answers.items():
        args: List[str] = query_schema["args"]
        qa_list: List[Tuple[List[int], Set[int]]] = query_schema["queries_answers"]
        for query, answer in qa_list:
            res.append((query_name, args, query, answer))
    return res


class TrainDataset(Dataset):
    def __init__(self, queries_answers: TYPE_train_queries_answers, nentity: int, nrelation: int, negative_sample_size: int):
        self.all_data: List[Tuple[str, List[str], List[int], Set[int]]] = flatten_train(queries_answers)
        random.shuffle(self.all_data)
        self.len: int = len(self.all_data)
        self.nentity: int = nentity
        self.nrelation: int = nrelation
        self.negative_sample_size: int = negative_sample_size
        self.count: Dict[str, int] = self.count_frequency(self.all_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, args, query, answer = self.all_data[idx]
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
        query = torch.LongTensor(query)  # (N,)
        #                                                                     (s, r, o, t) == (1, 7, 8, 5)
        # query_name         : str                                            'Pe'
        # args               : List[**str]                                    ['e1', 'r1', 't1']
        # query              : List[**int]                                    [ 1,    7,    5  ]
        # positive_answer    : torch.LongTensor (1,)                          Tensor([8])
        # negative_answer    : torch.LongTensor (self.negative_sample_size,)  Tensor([100, 392, 499, ...])
        # subsampling_weight : torch.FloatTensor (1,)                         Tensor([0.02])
        return query_name, args, query, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def collate_fn(data):
        query_name, args, query, positive_answer, negative_answer, subsampling_weight = tuple(zip(*data))
        query = torch.cat(query, dim=0)
        positive_answer = torch.cat(positive_answer, dim=0)
        negative_answer = torch.stack(negative_answer, dim=0)
        subsampling_weight = torch.cat(subsampling_weight, dim=0)
        return query_name, args, query, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def count_frequency(all_data: List[Tuple[str, List[str], List[int], Set[int]]], start=4) -> Dict[str, int]:
        count = {}
        for query_name, args, query, answer in all_data:
            count[query_name] = start + len(answer)
        return count


def flatten_test(queries_answers: TYPE_test_queries_answers) -> List[Tuple[str, List[str], List[int], Set[int], Set[int]]]:
    res = []
    for query_name, query_schema in queries_answers.items():
        args: List[str] = query_schema["args"]
        qa_list: List[Tuple[List[int], Set[int], Set[int]]] = query_schema["queries_answers"]
        for query, easy_answer, hard_answer in qa_list:
            res.append((query_name, args, query, easy_answer, hard_answer))
    return res


class TestDataset(Dataset):
    def __init__(self, queries_answers: TYPE_test_queries_answers, nentity, nrelation):
        self.all_data: List[Tuple[str, List[str], List[int], Set[int], Set[int]]] = flatten_test(queries_answers)
        random.shuffle(self.all_data)
        self.len: int = len(self.all_data)
        self.nentity: int = nentity
        self.nrelation: int = nrelation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, args, query, easy_answer, hard_answer = self.all_data[idx]
        candidate_answer = torch.LongTensor(range(self.nentity))
        hard_answer = set(hard_answer) - set(easy_answer)
        return candidate_answer, query_name, args, query, easy_answer, hard_answer

    @staticmethod
    def collate_fn(data):
        candidate_answer, query_name, args, query, easy_answer, hard_answer = tuple(zip(*data))
        query = torch.cat(query, dim=0)
        candidate_answer = torch.cat(candidate_answer, dim=0)
        return candidate_answer, query_name, args, query, easy_answer, hard_answer
