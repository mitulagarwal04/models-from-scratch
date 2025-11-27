import torch
import numpy as np


def load_weights_into_gpt(model, params):
    """Map TensorFlow GPT-2 checkpoint params → PyTorch model"""

    def assign(tensor, arr, transpose=False):
        arr = np.array(arr)
        if transpose:
            arr = arr.T
        t = torch.tensor(arr, dtype=tensor.dtype)
        if t.shape != tensor.shape:
            raise ValueError(f"Shape mismatch {tensor.shape} vs {t.shape}")
        tensor.data.copy_(t)

    # embeddings
    assign(model.tok_emb.weight, params["wte"])
    assign(model.pos_emb.weight, params["wpe"])

    # transformer blocks
    for i, block in enumerate(model.blocks):
        p = params["blocks"][i]

        # attention qkv
        assign(block.att.qkv.weight, p["attn"]["c_attn"]["w"], transpose=True)
        assign(block.att.qkv.bias, p["attn"]["c_attn"]["b"])

        # attention output proj
        assign(block.att.out_proj.weight, p["attn"]["c_proj"]["w"], transpose=True)
        assign(block.att.out_proj.bias, p["attn"]["c_proj"]["b"])

        # feed-forward
        assign(block.ff.net[0].weight, p["mlp"]["c_fc"]["w"], transpose=True)
        assign(block.ff.net[0].bias, p["mlp"]["c_fc"]["b"])

        assign(block.ff.net[3].weight, p["mlp"]["c_proj"]["w"], transpose=True)
        assign(block.ff.net[3].bias, p["mlp"]["c_proj"]["b"])

        # layer norms
        assign(block.norm1.scale, p["ln_1"]["g"])
        assign(block.norm1.shift, p["ln_1"]["b"])

        assign(block.norm2.scale, p["ln_2"]["g"])
        assign(block.norm2.shift, p["ln_2"]["b"])

    # final LN + head
    assign(model.final_norm.scale, params["g"])
    assign(model.final_norm.shift, params["b"])

    # lm_head tied to tok_emb
    assign(model.lm_head.weight, params["wte"])
