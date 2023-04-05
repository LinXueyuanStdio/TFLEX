from typing import List, Tuple, Set

import torch
from torch.utils.data import Dataset


class ComplementaryTrainDataset(Dataset):
    """
    生成 补集划分 的数据集
    head0: Bx(T-1)
    rel0: Bx(T-1)
    tail0: Bx(T-1)
    head: Bx1
    rel: Bx1
    tail: Bx1
    """

    def __init__(self, triples_ids: List[Tuple[int, int, int]]):
        self.triples_ids: List[Tuple[int, int, int]] = triples_ids
        self.triples_idx_set: Set[int] = set([i for i in range(len(triples_ids))])
        self.triples = torch.LongTensor(triples_ids)

    def __len__(self):
        return len(self.triples_ids)

    def __getitem__(self, idx):
        context_idx = list(self.triples_idx_set.difference({idx}))
        sample_triples = self.triples[idx]
        context_triples = self.triples[context_idx]
        head0, rel0, tail0 = context_triples[:, 0], context_triples[:, 1], context_triples[:, 2]
        head, rel, tail = sample_triples[0], sample_triples[1], sample_triples[2]
        return head0, rel0, tail0, head, rel, tail


class ComplementaryTestDataset(Dataset):
    """
    生成 补集划分 的数据集
    head: Bx1
    rel: Bx1
    tail: Bx1
    """

    def __init__(self, triples_ids: List[Tuple[int, int, int]]):
        self.triples_ids: List[Tuple[int, int, int]] = triples_ids
        self.triples_idx_set: Set[int] = set([i for i in range(len(triples_ids))])
        self.triples = torch.LongTensor(triples_ids)

    def __len__(self):
        return len(self.triples_ids)

    def __getitem__(self, idx):
        sample_triples = self.triples[idx]
        head, rel, tail = sample_triples[0], sample_triples[1], sample_triples[2]
        return head, rel, tail
