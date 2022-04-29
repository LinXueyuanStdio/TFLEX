from typing import List, Tuple, Dict, Set

import torch
from torch.utils.data import Dataset


class LinkPredictDataset(Dataset):
    def __init__(self, test_triples_ids: List[Tuple[int, int, int]], hr_t: Dict[Tuple[int, int], Set[int]], max_relation_id: int, entity_count: int):
        """
        test_triples_ids: without reverse r
        hr_t: all hr->t, MUST with reverse r
        """
        self.test_triples_ids = test_triples_ids
        self.hr_t = hr_t
        self.entity_count = entity_count
        self.max_relation_id = max_relation_id

    def __len__(self):
        return len(self.test_triples_ids)

    def __getitem__(self, idx):
        h, r, t = self.test_triples_ids[idx]
        reverse_r = r + self.max_relation_id

        mask_for_hr = torch.zeros(self.entity_count).long()
        mask_for_hr[list(self.hr_t[(h, r)])] = 1
        mask_for_hr[t] = 0

        mask_for_tReverser = torch.zeros(self.entity_count).long()
        mask_for_tReverser[list(self.hr_t[(t, reverse_r)])] = 1
        mask_for_tReverser[h] = 0

        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        reverse_r = torch.LongTensor([reverse_r])

        return h, r, mask_for_hr, t, reverse_r, mask_for_tReverser


class TemporalLinkPredictDataset(Dataset):
    def __init__(self, test_triples_ids: List[Tuple[int, int, int, int]], srt_o: Dict[Tuple[int, int, int], Set[int]], entity_count: int):
        """
        test_triples_ids: without reverse r
        hr_t: all hr->t, MUST with reverse r
        """
        self.test_triples_ids = test_triples_ids
        self.srt_o = srt_o
        self.entity_count = entity_count

    def __len__(self):
        return len(self.test_triples_ids)

    def __getitem__(self, idx):
        s, r, o, t = self.test_triples_ids[idx]

        mask_for_srt = torch.zeros(self.entity_count).long()
        mask_for_srt[list(self.srt_o[(s, r, t)])] = 1
        mask_for_srt[o] = 0

        s = torch.LongTensor([s])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        o = torch.LongTensor([o])
        return s, r, t, o, mask_for_srt


class LinkPredictTypeConstraintDataset(Dataset):
    def __init__(self,
                 test_triples_ids: List[Tuple[int, int, int]],
                 r_t: Dict[int, Set[int]],
                 hr_t: Dict[Tuple[int, int], Set[int]],
                 max_relation_id: int, entity_count: int):
        """
        test_triples_ids: without reverse r
        hr_t: all hr->t, MUST with reverse r
        r_t: all r->t, MUST with reverse r
        """
        self.test_triples_ids = test_triples_ids
        self.r_t = r_t
        self.hr_t = hr_t
        self.entity_count = entity_count
        self.max_relation_id = max_relation_id
        self.all_entity_ids = set(range(self.entity_count))

    def __len__(self):
        return len(self.test_triples_ids)

    def __getitem__(self, idx):
        h, r, t = self.test_triples_ids[idx]
        reverse_r = r + self.max_relation_id

        tail_type_constraint_mask_for_hr = torch.zeros(self.entity_count).long()
        valid = list({t} | (self.r_t[r] - self.hr_t[(h, r)]))
        tail_type_constraint_mask_for_hr[valid] = 1

        tail_type_constraint_for_tReverser = torch.zeros(self.entity_count).long()
        valid = list({h} | (self.r_t[reverse_r] - self.hr_t[(t, reverse_r)]))
        tail_type_constraint_for_tReverser[valid] = 1

        h = torch.LongTensor([h])
        r = torch.LongTensor([r])
        t = torch.LongTensor([t])
        reverse_r = torch.LongTensor([reverse_r])

        return h, r, tail_type_constraint_mask_for_hr, t, reverse_r, tail_type_constraint_for_tReverser
