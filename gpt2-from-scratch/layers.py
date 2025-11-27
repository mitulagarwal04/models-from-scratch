import torch
import torch.nn as nn
import numpy as np


# ----------------------------
# GELU
# ----------------------------
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))
    ))


# ----------------------------
# LayerNorm (GPT-2 style)
# ----------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normed + self.shift


# ----------------------------
# Causal Multi-head Attention
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, context_length=1024, qkv_bias=True, drop_rate=0.0):
        super().__init__()

        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.attn_drop = nn.Dropout(drop_rate)
        self.resid_drop = nn.Dropout(drop_rate)

        # Causal mask
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, T, hd)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        att = att.softmax(dim=-1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).reshape(B, T, C)

        out = self.out_proj(out)
        out = self.resid_drop(out)

        return out
