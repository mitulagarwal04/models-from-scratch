import torch.nn as nn
import torch
from layers_activations import silu, compute_rope, MultiHeadAttention, RMSNorm
from config import config

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = self.Linear(config['emb_dim '], config['hidden_dim'], dtype=config['dtype'], bais=False)
        self.fc2 = self.Linear(config['emb_dim '], config['hidden_dim'], dtype=config['dtype'], bais=False)
        self.fc3 = self.Linear(config['hidden_dim '], config['emb_dim'], dtype=config['dtype'], bais=False)
        self.silu = silu()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dtype=config["dtype"]
            # dropout=config["drop_rate"],
            # qkv_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config)

        self.norm1 = RMSNorm(config["emb_dim"])
        self.norm2 = RMSNorm(config["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.attn(x)

        return x + shortcut
    