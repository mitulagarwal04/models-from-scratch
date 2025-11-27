from transformers import GPT2TokenizerFast


def load_gpt2_tokenizer(model_dir="./gpt2/124M"):
    return GPT2TokenizerFast(
        vocab_file=f"{model_dir}/encoder.json",
        merges_file=f"{model_dir}/vocab.bpe",
    )
