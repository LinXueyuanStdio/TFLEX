"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/30
@description: null
"""
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class TripleDataset(Dataset):
    def __init__(self, triples_ids: List[Tuple[int, int, int]]):
        self.triples_ids = triples_ids

    def __len__(self):
        return len(self.triples_ids)

    def __getitem__(self, idx):
        h, r, t = self.triples_ids[idx]
        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        return h, r, t
