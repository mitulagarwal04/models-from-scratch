import torch
from layers_activations import compute_rope, precompute_rope_params


batch_size = 2
context_len = 5
num_heads = 4
head_dim = 16

cos, sin = precompute_rope_params(head_dim=head_dim, context_length=context_len)

torch.manual_seed(123)
queries = torch.randn(batch_size, num_heads, context_len, head_dim)
keys = torch.randn(batch_size, num_heads, context_len, head_dim)

queries_rot = compute_rope(queries, cos, sin)
keys_rot = compute_rope(keys, cos, sin)

print(queries_rot.shape)
print(queries_rot)

print(keys_rot.shape)
print(keys_rot) 

