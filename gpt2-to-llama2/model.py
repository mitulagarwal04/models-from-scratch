import torch
import torch.nn as nn
from config import config
from blocks import TransformerBlock
from layers_activations import RMSNorm

class LLama2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'], dtype=config['dtype'])
        
        self.trf_block - nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['layers'])]
        )

        self.final_norm = RMSNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False, dtype=config['dtype'])

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_block(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
