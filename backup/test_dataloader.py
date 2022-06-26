"""
@date: 2022/3/22
@description: null
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = "8888"
dist.init_process_group('nccl', rank=0, world_size=1)
torch.cuda.set_device(0)

class ToyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        query_name_idx = idx % 5 + 3
        len = query_name_idx + 3
        # query_name, args_idx, answer
        return f"{query_name_idx}", torch.LongTensor([i for i in range(len)]), torch.LongTensor([3])

    @staticmethod
    def collate_fn(data):
        d = {}
        target = torch.stack([_[2] for _ in data], dim=0)
        for name, vector, _ in data:
            d[name] = vector
        return d, target


dataset = ToyDataset()
val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
val_loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=4,
                                         sampler=val_sampler,
                                         collate_fn=ToyDataset.collate_fn)
for i in val_loader:
    print(i)
