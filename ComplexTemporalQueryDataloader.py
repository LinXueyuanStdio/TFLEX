"""
@date: 2022/3/16
@description: null
"""
from collections import defaultdict
from typing import List, Set, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from ComplexTemporalQueryData import TYPE_train_queries_answers, TYPE_test_queries_answers
from expression.TFLEX_DSL import is_to_predict_entity_set, query_structures


def flatten_train(queries_answers: TYPE_train_queries_answers) -> List[Tuple[str, List[int], Set[int]]]:
    res = []
    for query_name, query_schema in queries_answers.items():
        qa_list: List[Tuple[List[int], Set[int]]] = query_schema["queries_answers"]
        for query, answer in qa_list:
            res.append((query_name, query, answer))
    return res


class TrainDataset(Dataset):
    def __init__(self, queries_answers: TYPE_train_queries_answers, entity_count: int, timestamps_count: int, negative_sample_size: int):
        self.all_data: List[Tuple[str, List[int], Set[int]]] = flatten_train(queries_answers)
        self.len: int = len(self.all_data)
        self.entity_count: int = entity_count
        self.timestamps_count: int = timestamps_count
        self.negative_sample_size: int = negative_sample_size
        self.count: Dict[str, int] = self.count_frequency(self.all_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, query, answer = self.all_data[idx]
        tail = np.random.choice(list(answer))  # select one answer
        subsampling_weight = self.count[query_name]  # answer count of query
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # (1,)
        negative_sample_list = []
        negative_sample_size = 0
        answer_range = self.entity_count if is_to_predict_entity_set(query_name) else self.timestamps_count
        while negative_sample_size < self.negative_sample_size:
            negative_answer = np.random.randint(answer_range, size=self.negative_sample_size * 2)
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
        grouped_idxs: Dict[str, List[int]] = defaultdict(list)
        for i, (query_name, query, _, _, _) in enumerate(data):
            batch_queries_dict[query_name].append(query)
            grouped_idxs[query_name].append(i)
        grouped_query: Dict[str, torch.Tensor] = {
            key: torch.LongTensor(batch_queries_dict[key])
            for key in batch_queries_dict
        }
        return grouped_query, grouped_idxs, positive_answer, negative_answer, subsampling_weight

    @staticmethod
    def collate_fn2(data):
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
    def __init__(self, queries_answers: TYPE_test_queries_answers, entity_count: int, timestamps_count: int):
        self.all_data: List[Tuple[str, List[int], Set[int], Set[int]]] = flatten_test(queries_answers)
        self.len: int = len(self.all_data)
        self.entity_count: int = entity_count
        self.timestamps_count: int = timestamps_count

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query_name, query, easy_answer, hard_answer = self.all_data[idx]
        # query = torch.LongTensor(query)  # (N,)
        if len(easy_answer) >= len(hard_answer):
            easy_answer = set()
        hard_answer = set(hard_answer) - set(easy_answer)
        answer_range = self.entity_count if is_to_predict_entity_set(query_name) else self.timestamps_count
        candidate_answer = torch.LongTensor(range(answer_range))
        easy_answer_vector = torch.zeros(answer_range)
        if len(easy_answer) > 0:
            easy_answer_vector[list(easy_answer)] = 1
        easy_answer_mask = easy_answer_vector == 1
        return query_name, query, candidate_answer, easy_answer_mask, hard_answer

    @staticmethod
    def collate_fn(data):
        batch_queries_idx_dict: Dict[str, List[List[int]]] = defaultdict(list)
        batch_candidate_answer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        grouped_easy_answer_dict: Dict[str, List[torch.Tensor]] = defaultdict(list)
        grouped_hard_answer_dict: Dict[str, List[Set[int]]] = defaultdict(list)
        for i, (query_name, query, candidate_answer, easy_answer, hard_answer) in enumerate(data):
            if None in query:
                print("error", query_name, query)
            batch_queries_idx_dict[query_name].append(query)
            batch_candidate_answer_dict[query_name].append(candidate_answer)
            grouped_easy_answer_dict[query_name].append(easy_answer)
            grouped_hard_answer_dict[query_name].append(hard_answer)

            # in FLEX, it has used DNF for union
            # here we only cope with DM
            key_DM = f"{query_name}_DM"
            if key_DM in query_structures:
                batch_queries_idx_dict[key_DM].append(query)
                batch_candidate_answer_dict[key_DM].append(candidate_answer)
                grouped_easy_answer_dict[key_DM].append(easy_answer)
                grouped_hard_answer_dict[key_DM].append(hard_answer)

        grouped_query: Dict[str, torch.Tensor] = {
            key: torch.LongTensor(batch_queries_idx_dict[key])
            for key in batch_queries_idx_dict
        }
        grouped_candidate_answer: Dict[str, torch.Tensor] = {
            key: torch.stack(batch_candidate_answer_dict[key], dim=0)
            for key in batch_candidate_answer_dict
        }
        grouped_easy_answer: Dict[str, torch.Tensor] = {
            key: torch.stack(grouped_easy_answer_dict[key], dim=0)
            for key in grouped_easy_answer_dict
        }
        return grouped_query, grouped_candidate_answer, grouped_easy_answer, grouped_hard_answer_dict

    @staticmethod
    def collate_fn2(data):
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

            # in FLEX, it has used DNF for union
            # here we only cope with DM
            key_DM = f"{query_name}_DM"
            if key_DM in query_structures:
                grouped_query[key_DM].append(query)
                grouped_candidate_answer[key_DM].append(candidate_answer)
                grouped_easy_answer_dict[key_DM].append(easy_answer)
                grouped_hard_answer_dict[key_DM].append(hard_answer)

        grouped_easy_answer: Dict[str, torch.Tensor] = {
            key: torch.stack(grouped_easy_answer_dict[key], dim=0)
            for key in grouped_easy_answer_dict
        }
        data_list = []
        for query_name in grouped_query:
            data_list.append((query_name, torch.LongTensor(grouped_query[query_name]), torch.stack(grouped_candidate_answer[query_name], dim=0)))
        return data_list, grouped_easy_answer, grouped_hard_answer_dict
