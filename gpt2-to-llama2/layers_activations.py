import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emd_dim = emb_dim
        self.weights = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.sqrt(means + self.eps)
        return (x_normed * self.weights).to(dtype=x.dtype)
    

torch.manual_seed(123)

example_batch = torch.randn(2, 3, 4)

rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
print(rmsnorm_pytorch)
print(rms_norm)
# assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))