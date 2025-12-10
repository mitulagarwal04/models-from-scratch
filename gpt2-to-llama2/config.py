import torch

def config():
    return {
    "vocab_size": 32000,     
    "context_length": 4096,  
    "emb_dim": 4096,         
    "n_heads": 32,           
    "n_layers": 32,          
    "hidden_dim": 11008,     
    "dtype": torch.bfloat16  
}