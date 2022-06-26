"""
@date: 2021/10/27
@description: null
"""
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import Parameter


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # TODO key_padding_mask

        # print("[MultiheadAttention forward]")
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        # projecting q, k, v
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        # scaling q
        q *= self.scaling

        # extending k, v by one time step at the end, with self.bias_k and self.bias_v
        if self.bias_k is not None:
            assert self.bias_v is not None

            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # extending k, v by another time step at the end, (bsz * num_heads, 1, head_dim) of zeros
        if self.add_zero_attn:

            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights *= attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = (attn_weights - torch.min(attn_weights)) / (torch.max(attn_weights) - torch.min(attn_weights))
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    make_positions.range_buf = tensor.new().type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    mask = tensor.ne(padding_idx).to(positions.device)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().to(positions.device).masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        positions = make_positions(input, self.padding_idx, self.left_pad).to(self.weights.device)
        return self.weights.index_select(0, positions.view(-1)).to(self.weights.device).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, attn_mask=False):
        super().__init__()
        self.dropout = 0.3  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_A, input_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        input_A = self.scale_embed_position_dropout(input_A)
        input_B = self.scale_embed_position_dropout(input_B)
        # For each transformer encoder layer:
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B)
        return input_A, input_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True
        )
        self.attn_mask = attn_mask
        self.crossmodal = True
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = ComplexLinear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = ComplexLinear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        self.layer_norms_B = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_A, x_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        ## Attention Part
        # Residual and Layer Norm
        residual_A = x_A
        residual_B = x_B

        # Multihead Attention
        x_aaa = self.attention_block(x_A, x_A, x_A)
        x_aab = self.attention_block(x_A, x_A, x_B)
        x_aba = self.attention_block(x_A, x_B, x_A)
        x_baa = self.attention_block(x_B, x_A, x_A)
        x_abb = self.attention_block(x_A, x_B, x_B)
        x_bab = self.attention_block(x_B, x_A, x_B)
        x_bba = self.attention_block(x_B, x_B, x_A)
        x_bbb = self.attention_block(x_B, x_B, x_B)

        x_A = x_aaa - x_abb - x_bab - x_bba
        x_B = -x_bbb + x_baa + x_aba + x_aab

        x_A = self.layer_norms_A[0](x_A)
        x_B = self.layer_norms_B[0](x_B)
        # Dropout and Residual
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A = residual_A + x_A
        x_B = residual_B + x_B

        # ##FC Part
        residual_A = x_A
        residual_B = x_B

        # FC1
        x_A, x_B = self.fc1(x_A, x_B)
        x_A = F.relu(x_A)
        x_B = F.relu(x_B)
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)

        # FC2
        x_A, x_B = self.fc2(x_A, x_B)

        x_A = self.layer_norms_A[1](x_A)
        x_B = self.layer_norms_B[1](x_B)

        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A = residual_A + x_A
        x_B = residual_B + x_B

        return x_A, x_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):
        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, src_attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, tgt_attn_dropout=0.0):
        super().__init__()
        self.dropout = 0.3  # Embedding dropout
        # self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    src_attn_dropout=src_attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    tgt_attn_dropout=tgt_attn_dropout
                                    )
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_A, input_B, enc_A, enc_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
            enc_A: real part of encoder output.
            enc_B: imaginary part of encoder output.
        """
        input_A = self.scale_embed_position_dropout(input_A)
        input_B = self.scale_embed_position_dropout(input_B)

        # For each transformer encoder layer:
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B, enc_A, enc_B)
        return input_A, input_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, src_attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, tgt_attn_dropout=0.1, src_mask=True, tgt_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=src_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
        )
        self.src_mask = src_mask  # used as last arg in forward function call
        self.tgt_mask = tgt_mask  # used as last arg in forward function call

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        self.attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=tgt_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
        )

        self.fc1 = ComplexLinear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = ComplexLinear(self.embed_dim, self.embed_dim)

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])
        self.layer_norms_B = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def forward(self, x_A, x_B, enc_A, enc_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
            enc_A: real part of encoder output.
            enc_B: imaginary part of encoder output.
        """
        ## Attention Part
        # Residual and Layer Norm
        residual_A = x_A
        residual_B = x_B

        # Self Attention
        if self.src_mask:
            assert x_A.shape[0] == x_B.shape[0]
            mask = buffered_future_mask(x_A)
        else:
            mask = None

        x_aaa, _ = self.self_attn(x_A, x_A, x_A, attn_mask=mask)
        x_aab, _ = self.self_attn(x_A, x_A, x_B, attn_mask=mask)
        x_aba, _ = self.self_attn(x_A, x_B, x_A, attn_mask=mask)
        x_baa, _ = self.self_attn(x_B, x_A, x_A, attn_mask=mask)
        x_abb, _ = self.self_attn(x_A, x_B, x_B, attn_mask=mask)
        x_bab, _ = self.self_attn(x_B, x_A, x_B, attn_mask=mask)
        x_bba, _ = self.self_attn(x_B, x_B, x_A, attn_mask=mask)
        x_bbb, _ = self.self_attn(x_B, x_B, x_B, attn_mask=mask)

        x_A = x_aaa - x_abb - x_bab - x_bba
        x_B = -x_bbb + x_baa + x_aba + x_aab

        # Layer Norm, Dropout and Residual;
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        x_A += residual_A
        x_B += residual_B
        x_A = self.layer_norms_A[0](x_A)
        x_B = self.layer_norms_B[0](x_B)

        residual_A = x_A
        residual_B = x_B

        # Attention between encoder and decoder
        x_acc, _ = self.attn(x_A, enc_A, enc_A)
        x_add, _ = self.attn(x_A, enc_B, enc_B)
        x_bcd, _ = self.attn(x_B, enc_A, enc_B)
        x_bdc, _ = self.attn(x_B, enc_B, enc_A)
        x_acd, _ = self.attn(x_A, enc_A, enc_B)
        x_adc, _ = self.attn(x_A, enc_B, enc_A)
        x_bcc, _ = self.attn(x_B, enc_A, enc_A)
        x_bdd, _ = self.attn(x_B, enc_B, enc_B)

        x_A = x_acc - x_add - x_bcd - x_bdc
        x_B = x_acd + x_adc + x_bcc - x_bdd

        # Layer Norm, Dropout and Residual;
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        x_A += residual_A
        x_B += residual_B
        x_A = self.layer_norms_A[1](x_A)
        x_B = self.layer_norms_B[1](x_B)

        residual_A = x_A
        residual_B = x_B

        # FC1
        x_A, x_B = self.fc1(x_A, x_B)
        x_A = F.relu(x_A)
        x_B = F.relu(x_B)
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)

        # FC2
        x_A, x_B = self.fc2(x_A, x_B)
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A += residual_A
        x_B += residual_B
        x_A = self.layer_norms_A[2](x_A)
        x_B = self.layer_norms_B[2](x_B)
        # print("Attention here: ", x_A.mean().item(), x_B.mean().item())

        return x_A, x_B


class TransformerConcatEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, attn_mask=False):
        super().__init__()
        self.dropout = 0.3  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerConcatEncoderLayer(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          attn_dropout=attn_dropout,
                                          relu_dropout=relu_dropout,
                                          res_dropout=res_dropout,
                                          attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, x):
        x = self.scale_embed_position_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerConcatEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True
        )
        self.attn_mask = attn_mask
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        ## Attention Part
        # Residual and Layer Norm
        residual = x
        # Multihead Attention
        x = self.attention_block(x, x, x)

        x = self.layer_norms[0](x)
        # Dropout and Residual
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        # ##FC Part
        residual = x

        # FC1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)

        x = self.fc2(x)
        x = self.layer_norms[1](x)

        x = F.dropout(x, p=self.res_dropout, training=self.training)

        x = residual + x

        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):
        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x


class TransformerConcatDecoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, src_attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, tgt_attn_dropout=0.0):
        super().__init__()
        self.dropout = 0.3
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerConcatDecoderLayer(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          src_attn_dropout=src_attn_dropout,
                                          relu_dropout=relu_dropout,
                                          res_dropout=res_dropout,
                                          tgt_attn_dropout=tgt_attn_dropout
                                          )
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input, enc):
        input = self.scale_embed_position_dropout(input)
        for layer in self.layers:
            input = layer(input, enc)
        return input

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            y = self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)  # may change
        return x


class TransformerConcatDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, src_attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, tgt_attn_dropout=0.1, src_mask=True, tgt_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=src_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
        )
        self.src_mask = src_mask  # used as last arg in forward function call
        self.tgt_mask = tgt_mask  # used as last arg in forward function call

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        self.attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=tgt_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
        )

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def forward(self, x, enc):
        residual = x
        # Self Attention
        if self.src_mask:
            mask = buffered_future_mask(x)
        else:
            mask = None
        x, _ = self.self_attn(x, x, x)
        # Layer Norm, Dropout and Residual;
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x += residual
        x = self.layer_norms[0](x)

        residual = x

        # Attention between encoder and decoder
        x, _ = self.attn(x, enc, enc)

        # Layer Norm, Dropout and Residual;
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x += residual
        x = self.layer_norms[1](x)

        residual = x

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)

        x += residual
        x = self.layer_norms[2](x)

        return x


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def fill_with_one(t):
    return t.float().fill_(float(1)).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.tril(fill_with_one(torch.ones(dim1, dim2)), 0)
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim, eps=1e-20)
    return m


# including fnn, fnn_complex, fnn_crelu, rnn, lstm, gru

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)

        # x: (batch_size, seq_len, input_size)

    def forward(self, x):
        r_out, h_n = self.rnn(x, None)  # r_out: (batch_size, seq_len, hidden_size)
        last_time_step_out = r_out[:, -1, :]
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))

        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)

        # x: (batch_size, seq_len, input_size)

    def forward(self, x):
        r_out, h_n = self.gru(x, None)  # r_out: (batch_size, seq_len, hidden_size)

        last_time_step_out = r_out[:, -1, :]
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))

        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_size, output_size, bidirectional, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)

        # x: (batch_size, seq_len, input_size)

    def forward(self, x):
        r_out, h_n = self.rnn(x, None)  # r_out: (batch_size, seq_len, hidden_size)

        last_time_step_out = r_out[:, -1, :]
        last_layer_output = self.fc2(F.relu(self.fc1(last_time_step_out)))

        return r_out, nn.functional.log_softmax(last_layer_output, dim=1)


def eval_RNN_Model(data_loader, time_step, input_size, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()

        total_loss = 0
        total_pred = 0
        total_correct = 0
        for data_batched, label_batched in data_loader:
            # prepare inputs
            cur_batch_size = len(data_batched)
            data_batched = data_batched.reshape(cur_batch_size, time_step, input_size)  # (batch_size, feature_dim) -> (batch_size, time_step, input_size)
            label_batched = np.argmax(label_batched, axis=1).long()

            data_var = data_batched.float().to(device=device)
            label_var = label_batched.view(-1).to(device=device)  # (batch_size)

            label_np = np.asarray(label_batched)

            # run model and eval
            compressed_signal, output = model(data_var)
            loss = loss_func(output, label_var).item()
            pred = np.argmax(output.data.cpu().numpy(), axis=1)

            total_loss += loss
            total_pred += cur_batch_size
            total_correct += (pred == label_np).sum()

        acc = float(total_correct) / float(total_pred)
        print("%s loss %f and acc %f " % (name, total_loss, acc))

    return None, total_loss, acc


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()
        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(self.num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.bn.append(nn.BatchNorm1d(hidden_size[i + 1]))
        self.fc2 = nn.Linear(hidden_size[self.num_hidden - 1], output_size)

    def forward(self, x):
        global device
        var_x = x.to(device=device)  # (100, 20, 160)
        logitis = self.dropout(self.non_linear(self.bn1(self.fc1(var_x))))
        for i in range(self.num_hidden - 1):
            logitis = self.dropout(self.non_linear(self.bn[i](self.hidden[i](logitis))))
        compressed_signal = logitis
        logitis = self.fc2(logitis)
        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)


class FNN_crelu(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, non_linear='tanh', dropout=0.0):
        super(FNN_crelu, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = len(hidden_size)
        self.non_linear = nn.ReLU()

        self.fc1_w_real = nn.Linear(input_size // 2, hidden_size[0] // 2, bias=False)
        self.fc1_w_imag = nn.Linear(input_size // 2, hidden_size[0] // 2, bias=False)
        self.fc1_b = torch.Tensor(np.random.randn(hidden_size[0]) / np.sqrt(hidden_size[0])).to(device=device)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        self.w_real = nn.ModuleList()
        self.w_imag = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.b = []
        for i in range(self.num_hidden - 1):
            self.w_real.append(nn.Linear(hidden_size[i] // 2, hidden_size[i + 1] // 2, bias=False))
            self.w_imag.append(nn.Linear(hidden_size[i] // 2, hidden_size[i + 1] // 2, bias=False))
            self.bn.append(nn.BatchNorm1d(hidden_size[i + 1]))
            self.b.append(torch.Tensor(np.random.randn(hidden_size[i + 1]) / np.sqrt(hidden_size[i + 1])).to(device=device))

        self.fc2_w_real = nn.Linear(hidden_size[self.num_hidden - 1] // 2, output_size // 2, bias=False)
        self.fc2_w_imag = nn.Linear(hidden_size[self.num_hidden - 1] // 2, output_size // 2, bias=False)
        self.fc2_b = torch.Tensor(np.random.randn(output_size) / np.sqrt(output_size)).to(device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        global device
        x = x.to(device=device)

        even_indices = torch.tensor([i for i in range(self.input_size) if i % 2 == 0]).to(device=device)
        odd_indices = torch.tensor([i for i in range(self.input_size) if i % 2 == 1]).to(device=device)

        real_input = torch.index_select(x, 1, even_indices)  # (bs, input_size/2)
        imag_input = torch.index_select(x, 1, odd_indices)  # (bs, input_size/2)
        up = self.fc1_w_real(real_input) - self.fc1_w_imag(imag_input)
        down = self.fc1_w_imag(real_input) + self.fc1_w_real(imag_input)
        logitis = torch.cat((up, down), dim=1)
        logitis += self.fc1_b
        logitis = self.dropout(self.non_linear(self.bn1(logitis)))

        for i in range(self.num_hidden - 1):
            real_input = logitis[:, :self.hidden_size[i] // 2]
            imag_input = logitis[:, self.hidden_size[i] // 2:]
            up = self.w_real[i](real_input) - self.w_imag[i](imag_input)
            down = self.w_imag[i](real_input) + self.w_real[i](imag_input)
            logitis = torch.cat((up, down), dim=1)
            logitis += self.b[i]
            logitis = self.dropout(self.non_linear(self.bn[i](logitis)))

        compressed_signal = logitis

        var_real_input = logitis[:, :self.hidden_size[self.num_hidden - 1] // 2]
        var_imag_input = logitis[:, self.hidden_size[self.num_hidden - 1] // 2:]

        up = self.fc2_w_real(var_real_input) - self.fc2_w_imag(var_imag_input)
        down = self.fc2_w_imag(var_real_input) + self.fc2_w_real(var_imag_input)
        logitis = torch.cat((up, down), dim=1)
        logitis += self.fc2_b
        logitis = self.non_linear(logitis)

        return compressed_signal, nn.functional.log_softmax(logitis, dim=1)


class ComplexSequential(nn.Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)

    def forward(self, input_r, input_i):
        return self.dropout_r(input_r), self.dropout_i(input_i)


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self, input_r, input_i):
        return self.relu_r(input_r), self.relu_i(input_i)


class ComplexConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        return self.conv_r(input_r) - self.conv_i(input_i), \
               self.conv_r(input_i) + self.conv_i(input_r)


class ComplexMaxPool1d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.maxpool_r = nn.MaxPool1d(kernel_size=self.kernel_size,
                                      stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, ceil_mode=self.ceil_mode,
                                      return_indices=self.return_indices)
        self.maxpool_i = nn.MaxPool1d(kernel_size=self.kernel_size,
                                      stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, ceil_mode=self.ceil_mode,
                                      return_indices=self.return_indices)

    def forward(self, input_r, input_i):
        return self.maxpool_r(input_r), self.maxpool_i(input_i)


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), \
               self.fc_r(input_i) + self.fc_i(input_r)


class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        shape = input_r.shape
        input_r = input_r.reshape(-1, shape[1])
        input_i = input_i.reshape(-1, shape[1])

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            self.running_mean = exponential_average_factor * mean \
                                + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input_r = input_r - mean_r[None, :]
            input_i = input_i - mean_i[None, :]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = input_r.var(dim=0, unbiased=False) + self.eps
            Cii = input_i.var(dim=0, unbiased=False) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=0)

            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]
            # zero mean values
            input_r = input_r - mean[None, :, 0]
            input_i = input_i - mean[None, :, 1]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :] * input_r + Rri[None, :] * input_i, \
                           Rii[None, :] * input_i + Rri[None, :] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0] * input_r + self.weight[None, :, 2] * input_i + \
                               self.bias[None, :, 0], \
                               self.weight[None, :, 2] * input_r + self.weight[None, :, 1] * input_i + \
                               self.bias[None, :, 1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        input_r = input_r.reshape(shape)
        input_i = input_i.reshape(shape)

        return input_r, input_i


class ComplexFlatten(nn.Module):
    def forward(self, input_r, input_i):
        input_r = input_r.view(input_r.size()[0], -1)
        input_i = input_i.view(input_i.size()[0], -1)
        return input_r, input_i


def eval_FNN(data, label, model, num_classes, loss_func, name, path):
    global device
    with torch.no_grad():
        model.eval()

        data = torch.from_numpy(data).float().to(device=device)
        true_label = np.argmax(label, axis=1)
        label = torch.from_numpy(true_label).long().view(-1).to(device=device)  # -1
        compressed_signal, output = model(data)
        output = output.view(-1, num_classes)
        l = loss_func(output, label).item()
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        acc = np.mean(pred == true_label.reshape(-1))
        print("%s loss %f and acc %f " % (name, l, acc))

        # Confusion Matrix Calculator
        cnf_matrix = confusion_matrix(true_label.reshape(-1), pred)

        # Normalize Confusion Matrix
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        save_path = os.path.join(path, 'confusion_matrix_' + name)
        np.save(save_path, cnf_matrix)
    return compressed_signal, l, acc


class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, input_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size, input_dim]
        src = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(src)

        return hidden, cell


class Decoder_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = input_dim
        self.final_output_dim = output_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)  # inserting a new dimension to be seq len
        input = self.dropout(self.fc1(input))
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc2(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.out_fc = nn.Linear(decoder.output_dim, decoder.final_output_dim)
        self.final_fc = nn.Linear(decoder.final_output_dim, 1000)
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, dataset, teacher_forcing_ratio=0.0):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_input_size = trg.shape[2]
        trg_output_dim = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_output_dim).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = torch.zeros(batch_size, trg_input_size).to(self.device)

        for t in range(max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)  # output: [batch size, output dim]
            outputs[t] = output  # storing ouput at: 1 -> max_len
            teacher_force = random.random() < teacher_forcing_ratio
            input = (trg[t] if teacher_force else output)
        outputs = self.out_fc(outputs)
        if dataset == "iq":
            outputs = self.final_fc(outputs[-1])
        return outputs  # (seq_len, bs, output_dim)


def eval_Seq2Seq(data_loader, src_time_step, trg_time_step, input_size, model, criterion, name, path, device, dataset, dataset_raw):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for data_batched, label_batched in data_loader:
            cur_batch_size = len(data_batched)
            src = data_batched[:, 0: src_time_step, :].transpose(1, 0).float().cuda()
            trg = data_batched[:, src_time_step:, :].transpose(1, 0).float().cuda()
            trg_label = label_batched.cuda()
            outputs = model(src=src, trg=trg, dataset=dataset)  # (ts, bs, input_size)
            if dataset == "music":
                trg_label = label_batched[:, src_time_step:, :].transpose(1, 0).cuda()
                loss = criterion(outputs.transpose(0, 1).double(), trg_label.transpose(0, 1).double())
            elif dataset == "iq":
                loss = criterion(outputs.double(), trg_label.long())
            epoch_loss += loss.detach().item()
        if dataset == "music":
            avg_loss = epoch_loss / float(len(data_loader))
        elif dataset == "iq":
            avg_loss = epoch_loss / float(len(dataset_raw))
        print("%s loss %f" % (name, avg_loss))
    return avg_loss


if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
