def get_gpt2_config(settings):
    return {
        "vocab_size": settings.get("vocab_size", 50257),
        "context_length": settings.get("n_ctx", 1024),
        "emb_dim": settings.get("n_embd", 768),
        "n_layers": settings.get("n_layer", 12),
        "n_heads": settings.get("n_head", 12),
        "drop_rate": 0.0,
        "qkv_bias": True
    }
