"""
@date: 2022/4/14
@description: null
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreConvE(nn.Module):
    def __init__(self, embedding_dim, img_h=10, input_dropout=0.2, hidden_dropout1=0.3, hidden_dropout2=0.2):
        super(CoreConvE, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.feature_map_drop = nn.Dropout2d(hidden_dropout1)
        self.hidden_drop = nn.Dropout(hidden_dropout2)

        self.img_h = img_h
        self.img_w = embedding_dim // self.img_h

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        hidden_size = (self.img_h * 3 - 3 + 1) * (self.img_w - 3 + 1) * 32
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, s, r, t):
        s = s.view(-1, 1, self.img_h, self.img_w)
        r = r.view(-1, 1, self.img_h, self.img_w)
        t = t.view(-1, 1, self.img_h, self.img_w)

        x = torch.cat([s, r, t], dim=2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    B = 10
    d = 200
    q = torch.rand((B, d))
    r = torch.rand((B, d))
    t = torch.rand((B, d))
    core = CoreConvE(d)
    res = core(q, r, t)
    print(res.shape)
