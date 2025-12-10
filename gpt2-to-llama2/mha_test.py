from layers_activations import MultiHeadAttention
import torch

batch_size = 1
context_len = 100
max_context_len = 4096
embed_dim = 128
num_heads = 4


example_batch = torch.randn((batch_size, context_len, embed_dim))

mha = MultiHeadAttention(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=max_context_len,
    num_heads=num_heads
)

print(mha(example_batch))

del mha