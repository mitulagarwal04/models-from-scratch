from huggingface_hub import login
from huggingface_hub import hf_hub_download
import sentencepiece as spm
import json
import os
from dotenv import load_dotenv

load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    access_token = config[HF_ACCESS_TOKEN]

login(token = access_token)


tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="./Llama2/Llama-2-7b"
)

class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


tokenizer = LlamaTokenizer(tokenizer_file)