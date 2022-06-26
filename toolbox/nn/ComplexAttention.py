"""
@date: 2021/10/27
@description: null
"""

import torch
from torch import nn


class ComplexMatrixMult(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    """

    def forward(self, x, W):
        x_a, x_b = x
        W_a, W_b = W
        if len(x_a) > 2:
            """
            Shapes for inputs:
                - x: :math:`(L, N, E)` where L is the sequence length, N is the batch size, E is the embedding dimension.
                - W: :math:`(L, N, E)`, where S is the sequence length, N is the batch size, E is the embedding dimension.
            """
            out_a = x_a.bmm(W_a) - x_b.bmm(W_b)
            out_b = x_b.bmm(W_a) + x_a.bmm(W_b)
        else:
            """
            Shapes for inputs:
                - x: :math:`(N, E)` where N is the batch size, E is the embedding dimension.
                - W: :math:`(N, E)`, where N is the batch size, E is the embedding dimension.
            """
            out_a = x_a.mm(W_a) - x_b.mm(W_b)
            out_b = x_b.mm(W_a) + x_a.mm(W_b)
        return out_a, out_b


class ComplexLinear(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    out in C^d_out, out_a in (B, d_out), out_b in (B, d_out)
    """

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.W_a = nn.Linear(in_features, out_features)
        self.W_b = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_a, x_b = x
        out_a = self.W_a(x_a) - self.W_b(x_b)
        out_b = self.W_a(x_b) + self.W_b(x_a)
        return out_a, out_b


class ComplexRealLinear(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    out in C^d_out, out_a in (B, d_out), out_b in (B, d_out)
    """

    def __init__(self, in_features, out_features):
        super(ComplexRealLinear, self).__init__()
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_a, x_b = x
        out_a = self.W(x_a)
        out_b = self.W(x_b)
        return out_a, out_b


class ComplexAttention(nn.Module):
    """
    X = X_a + X_b i
    Q = X * W_q = (X_a + X_b i) * Wq = (X_a * Wq) + (X_b * Wq) i
    K = X * W_k = (X_a + X_b i) * Wk = (X_a * Wk) + (X_b * Wk) i
    V = X * W_v = (X_a + X_b i) * Wv = (X_a * Wv) + (X_b * Wv) i
    Attn(X) = Q * K^T * V
    X in C^d, X_a in (B, d), X_b in (B, d)
    """

    def __init__(self):
        super(ComplexAttention, self).__init__()
        self.matrixMul = ComplexMatrixMult()

    def forward(self, Q, K, V):
        K_a, K_b = K  # (B, d_out) (B, d_out)
        K_T = (K_a.transpose(-1, -2), K_b.transpose(-1, -2))  # (d_out, B) (d_out, B)
        out = self.matrixMul(self.matrixMul(Q, K_T), V)  # (B, d_out) (B, d_out)
        return out


class ComplexSelfAttention(nn.Module):
    """
    X = X_a + X_b i
    Q = X * W_q = (X_a + X_b i) * (Wq_a + Wq_b i) = (X_a * Wq_a - X_b * Wq_b) + (X_b * Wq_a + X_a * Wq_b) i
    K = X * W_k = (X_a + X_b i) * (Wk_a + Wk_b i) = (X_a * Wk_a - X_b * Wk_b) + (X_b * Wk_a + X_a * Wk_b) i
    K^T = (X_a * Wk_a - X_b * Wk_b)^T + (X_b * Wk_a + X_a * Wk_b)^T i
    V = X * W_v = (X_a + X_b i) * (Wv_a + Wv_b i) = (X_a * Wv_a - X_b * Wv_b) + (X_b * Wv_a + X_a * Wv_b) i
    Attn(X) = Q * K^T * V
    X in C^d, X_a in (B, d), X_b in (B, d)
    """

    def __init__(self, in_features, out_features):
        super(ComplexSelfAttention, self).__init__()
        self.Wq = ComplexLinear(in_features, out_features)
        self.Wk = ComplexLinear(in_features, out_features)
        self.Wv = ComplexLinear(in_features, out_features)
        self.attn = ComplexAttention()

    def forward(self, X):
        """X is (B, d) (B, d)"""
        Q = self.Wq(X)  # (B, d_out) (B, d_out)
        K = self.Wk(X)  # (B, d_out) (B, d_out)
        V = self.Wv(X)  # (B, d_out) (B, d_out)
        out = self.attn(Q, K, V)  # (B, d_out) (B, d_out)
        return out


class ComplexRealSelfAttention(nn.Module):
    """
    X = X_a + X_b i
    Q = X * W_q = (X_a + X_b i) * Wq = (X_a * Wq) + (X_b * Wq) i
    K = X * W_k = (X_a + X_b i) * Wk = (X_a * Wk) + (X_b * Wk) i
    V = X * W_v = (X_a + X_b i) * Wv = (X_a * Wv) + (X_b * Wv) i
    Attn(X) = Q * K^T * V
    X in C^d, X_a in (B, d), X_b in (B, d)
    """

    def __init__(self, in_features, out_features):
        super(ComplexRealSelfAttention, self).__init__()
        self.Wq = ComplexRealLinear(in_features, out_features)
        self.Wk = ComplexRealLinear(in_features, out_features)
        self.Wv = ComplexRealLinear(in_features, out_features)
        self.attn = ComplexAttention()

    def forward(self, X):
        """X is (B, d) (B, d)"""
        Q = self.Wq(X)  # (B, d_out) (B, d_out)
        K = self.Wk(X)  # (B, d_out) (B, d_out)
        V = self.Wv(X)  # (B, d_out) (B, d_out)
        out = self.attn(Q, K, V)  # (B, d_out) (B, d_out)
        return out


if __name__ == '__main__':
    B = 5
    L = 8
    d = 10
    d_out = 15
    X = (torch.rand((L, B, d)), torch.rand((L, B, d)))
    attn = ComplexSelfAttention(d, d_out)
    out_a, out_b = attn(X)
    print(out_a.shape, out_b.shape)
    # torch.Size([5, 15]) torch.Size([5, 15])
    attn2 = ComplexRealSelfAttention(d, d_out)
    out_a, out_b = attn2(X)
    print(out_a.shape, out_b.shape)
    # torch.Size([5, 15]) torch.Size([5, 15])
