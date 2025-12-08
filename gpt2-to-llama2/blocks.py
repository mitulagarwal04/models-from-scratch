import torch.nn as nn
import torch
from layers_activations import silu, compute_rope

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = self.Linear(cfg['emb_dim '], cfg['hidden_dim'], dtype=cfg['dtype'], bais=False)
        self.fc2 = self.Linear(cfg['emb_dim '], cfg['hidden_dim'], dtype=cfg['dtype'], bais=False)
        self.fc3 = self.Linear(cfg['hidden_dim '], cfg['emb_dim'], dtype=cfg['dtype'], bais=False)
        self.silu = silu()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
