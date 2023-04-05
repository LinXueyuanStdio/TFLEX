from functools import reduce
from operator import mul
from typing import List

import torch.nn.functional as F
from torch import nn


class ContextualParameterGenerator(nn.Module):
    def __init__(self, feature_in_dim: int, shape: List[int]):
        super(ContextualParameterGenerator, self).__init__()
        self.feature_in_dim = feature_in_dim
        self.shape = shape
        self.feature_out_dim = reduce(mul, shape, 1)
        self.hidden_dim = 200
        self.generate = nn.Sequential(
            nn.Linear(self.feature_in_dim, self.hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_out_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.generate(x)
        return x.view(-1, *self.shape)


class CoreCoPER(nn.Module):
    def __init__(self,
                 feature_in_dim: int,
                 feature_out_dim: int,
                 generate_dim: int,
                 hidden_dropout1: float = 0.2,
                 hidden_dropout2: float = 0.2):
        super(CoreCoPER, self).__init__()
        self.feature_in_dim = feature_in_dim
        self.feature_out_dim = feature_out_dim
        self.generate_dim = generate_dim

        self.conv_in_height = 10
        self.conv_in_width = self.feature_in_dim // 10

        self.conv_filter_height = 3
        self.conv_filter_width = 3
        self.conv_num_channels = 32

        self.conv_out_height = self.conv_in_height - self.conv_filter_height + 1
        self.conv_out_width = self.conv_in_width - self.conv_filter_width + 1

        self.fc_input_dim = self.conv_out_height * self.conv_out_width * self.conv_num_channels

        self.generate_conv_weight = ContextualParameterGenerator(self.generate_dim, [self.conv_num_channels, 1, self.conv_filter_height, self.conv_filter_width])  # conv weight
        self.generate_conv_bias = ContextualParameterGenerator(self.generate_dim, [self.conv_num_channels])  # conv bias
        self.generate_fc_weight = ContextualParameterGenerator(self.generate_dim, [self.fc_input_dim, self.feature_out_dim])  # fc weight
        self.generate_fc_bias = ContextualParameterGenerator(self.generate_dim, [1, self.feature_out_dim])  # fc bias

        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        self.bn1 = nn.BatchNorm1d(self.feature_out_dim)

        self.m = nn.PReLU()

    def forward(self, input_embedding, generate_embedding):
        """
        input_embedding: batch_size x self.feature_in_dim
        generate_embedding: batch_size x self.generate_dim
        """
        img = input_embedding.view(1, -1, self.conv_in_height, self.conv_in_width)

        r = generate_embedding
        batch_size = generate_embedding.size(0)

        conv_weight = self.generate_conv_weight(r).view(-1, 1, self.conv_filter_height, self.conv_filter_width)
        conv_bias = self.generate_conv_bias(r).view(-1)
        fc_weight = self.generate_fc_weight(r)
        fc_bias = self.generate_fc_bias(r)

        x = F.conv2d(img, conv_weight, bias=conv_bias, groups=batch_size)
        x = F.relu(x)
        x = self.hidden_dropout1(x)
        x = x.view(-1, 1, self.fc_input_dim)

        x = x.bmm(fc_weight) + fc_bias
        x = x.view(-1, self.feature_out_dim)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.m(x)
        return x
