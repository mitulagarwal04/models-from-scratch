import torch.nn as nn
from layers import LayerNorm, CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, emb_dim, mlp_dim, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, mlp_dim, context_length, drop_rate=0.0, qkv_bias=True):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.att = CausalSelfAttention(
            emb_dim, n_heads,
            context_length=context_length,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate
        )
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
