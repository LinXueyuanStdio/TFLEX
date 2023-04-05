import os
from typing import Optional, List

import torch
from torch import distributed as dist


class DistributeSchema:

    def __init__(self, local_rank=-1, gpus: Optional[List[int]] = None):
        self.local_rank = local_rank
        self.gpus = gpus
        self.world_size = self.get_world_size()
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")
        self.device = self.get_device()

    def reduce_sum(self, tensor):
        if self.get_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def running_in_main_node(self) -> bool:
        return self.get_local_rank() == 0

    def get_local_rank(self):
        if self.local_rank == -1:
            if dist.is_initialized():
                return dist.get_rank()
            if "RANK" in os.environ:
                return int(os.environ["RANK"])
            return 0
        return self.local_rank

    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        return 1

    def synchronize(self):
        if self.get_world_size() > 1:
            dist.barrier()

    def get_device(self):
        if self.gpus:
            device = torch.device(self.gpus[self.get_local_rank()])
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        return device

    def wrap_to_parallel_model(self, model):
        if self.world_size > 1:
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        return model

    def __repr__(self):
        return f"{self.__class__.__name__}(gpu={self.local_rank} in {self.gpus})"

