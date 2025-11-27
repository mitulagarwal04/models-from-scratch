import torch.nn as nn
import torch
from blocks import TransformerBlock
from layers import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vocab = cfg["vocab_size"]
        self.ctx = cfg["context_length"]
        emb = cfg["emb_dim"]

        self.tok_emb = nn.Embedding(self.vocab, emb)
        self.pos_emb = nn.Embedding(self.ctx, emb)
        self.drop = nn.Dropout(cfg["drop_rate"])

        mlp_dim = 4 * emb

        self.blocks = nn.ModuleList([
            TransformerBlock(
                emb, cfg["n_heads"], mlp_dim,
                context_length=self.ctx,
                drop_rate=cfg["drop_rate"]
            ) for _ in range(cfg["n_layers"])
        ])

        self.final_norm = LayerNorm(emb)
        self.lm_head = nn.Linear(emb, self.vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
