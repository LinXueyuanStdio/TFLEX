import random
from typing import List, Tuple, Set, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from toolbox.data.functional import build_map_hr_t


def get_neg_sampling_batch(entity_ids: Set[int],
                           hr_t: Dict[Tuple[int, int], Set[int]],
                           hr_pairs: List[Tuple[int, int]],
                           idx: int,
                           sampling_window_size=200):
    assert sampling_window_size in range(len(entity_ids))
    assert idx in range(len(hr_pairs))
    max_positive_count = sampling_window_size // 10
    batch = hr_pairs[idx]
    target_sim = np.zeros(sampling_window_size)
    target_ids = []
    positive_ids: List[int] = list(hr_t[batch])
    if len(positive_ids) > max_positive_count:
        positive_ids = random.choices(positive_ids, k=max_positive_count)
    positive_count = len(positive_ids)
    if positive_count >= sampling_window_size:
        ids = random.choices(positive_ids, k=sampling_window_size)
        target_ids.append(ids)
        target_sim[:] = 1.
    else:
        negative_count = sampling_window_size - positive_count
        negative_ids = list(entity_ids.difference(set(positive_ids)))
        negative_ids = random.choices(negative_ids, k=negative_count)
        ids = positive_ids + negative_ids
        target_ids.append(ids)
        target_sim[:positive_count] = 1.
    target_ids = torch.LongTensor(target_ids).view(-1)
    target_sim = torch.FloatTensor(target_sim).view(-1)
    return np.array(batch), target_ids, target_sim


class FixedWindowNegSamplingDataset(Dataset):
    def __init__(self, train_triples_ids: List[Tuple[int, int, int]], entity_ids: List[int], sampling_window_size=200):
        self.hr_t = build_map_hr_t(train_triples_ids)
        self.hr_pairs = list(self.hr_t.keys())
        self.sampling_window_size = sampling_window_size
        self.entity_ids: Set[int] = set(entity_ids)

    def __len__(self):
        return len(self.hr_pairs)

    def __getitem__(self, idx):
        return get_neg_sampling_batch(self.entity_ids, self.hr_t, self.hr_pairs, idx, self.sampling_window_size)
