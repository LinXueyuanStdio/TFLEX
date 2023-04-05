from typing import Tuple, Sequence

import torch
from torch import nn


class Fro(nn.Module):
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(torch.norm(f, 2) ** 2)
        return norm / factors[0][0].shape[0]


class N3(nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(torch.abs(f) ** 3) / f.shape[0]
        return norm


class L1(nn.Module):
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(torch.abs(f) ** 1)
        return norm / factors[0][0].shape[0]


class L2(nn.Module):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(torch.abs(f) ** 2)
        return norm / factors[0][0].shape[0]


class NA(nn.Module):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        return torch.Tensor([0.0]).cuda()


class DURA(nn.Module):
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0

        for factor in factors:
            h, r, t = factor

            norm += torch.sum(t ** 2 + h ** 2)
            norm += torch.sum(h ** 2 * r ** 2 + t ** 2 * r ** 2)

        return self.weight * norm / h.shape[0]


class DURA_RESCAL(nn.Module):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 2 + t ** 2)
            norm += torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]


class DURA_RESCAL_W(nn.Module):
    def __init__(self, weight: float):
        super(DURA_RESCAL_W, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += 2.0 * torch.sum(h ** 2 + t ** 2)
            norm += 0.5 * torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]


class DURA_W(nn.Module):
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors: Sequence[torch.Tensor]):
        norm = 0
        for factor in factors:
            h, r, t = factor

            norm += 0.5 * torch.sum(t ** 2 + h ** 2)
            norm += 1.5 * torch.sum(h ** 2 * r ** 2 + t ** 2 * r ** 2)

        return self.weight * norm / h.shape[0]
