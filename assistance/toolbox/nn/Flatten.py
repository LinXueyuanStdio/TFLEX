import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        n = x.size(0)
        x = x.view(n, -1)
        return x
