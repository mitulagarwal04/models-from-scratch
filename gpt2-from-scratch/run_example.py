import torch
import json
import numpy as np
from config import get_gpt2_config
from model import GPTModel
from load_gpt2 import load_weights_into_gpt
from tokenizer import load_gpt2_tokenizer
import warnings
warnings.filterwarnings("ignore")


# ---------------- Load settings ----------------
with open("./gpt2/124M/hparams.json", "r") as f:
    settings = json.load(f)

# Load parameters from TF checkpoint
import tensorflow as tf

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

params = load_gpt2_params_from_tf_ckpt(
    "./gpt2/124M/model.ckpt",
    settings    
)

# ---------------- Init model ----------------
cfg = get_gpt2_config(settings)
model = GPTModel(cfg)
model.eval()

load_weights_into_gpt(model, params)

# ---------------- Tokenizer ----------------
tokenizer = load_gpt2_tokenizer("./gpt2/124M")

# ---------------- Run inference ----------------
prompt = "Python is a"
num_tokens = 15
ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

with torch.no_grad():
    for _ in range(num_tokens):
        logits = model(ids)
        next_id = logits[:, -1].argmax(dim=-1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)

# Decode final output
print("Generated text:\n")
print(tokenizer.decode(ids[0]))
