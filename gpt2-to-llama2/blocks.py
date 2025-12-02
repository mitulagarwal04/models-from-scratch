import torch.nn as nn
import torch


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = self.Linear(cfg['emb_dim'], )
