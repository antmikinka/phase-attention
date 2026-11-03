"""Tiny random-init copy of the Qwen3.8-27B text model structure.

Real 27B (config.json of Qwen/Qwen3.8-27B, text_config):
  64 layers = 16 x (3 x linear_attention + 1 x full_attention), hidden 5120, FFN 17408, vocab 248320
  full attn: 24 Q heads / 4 KV heads (GQA 6:1), head_dim 256, attn_output_gate, partial_rotary_factor 0.25,
             interleaved MRoPE sections [11, 11, 10] (sum 32 = rotary_dim/2), rope_theta 1e7
  DeltaNet : 16 QK heads / 48 V heads (1:3), head dims 128, conv kernel 4
Tiny copy keeps every *ratio/semantics* and shrinks sizes:
  8 layers = 2 x (3 lin + 1 full), hidden 64, FFN 128, 6 Q heads / 1 KV head (GQA 6:1), head_dim 32,
  rotary dim 8 -> MRoPE sections [2, 1, 1] (sum 4), DeltaNet 2 QK / 6 V heads of dim 16.
The vocab stays at the real 248320 so the real Qwen3.8 tokenizer (and llama.cpp's converter) works unchanged.
"""
from __future__ import annotations
import os
import torch

TOKENIZER_DIR = os.environ.get("QWEN38_TOKENIZER_DIR", "/workspace/qwen38-tok")


def tiny_config(n_blocks: int = 2, mtp: bool = False, init_range: float = 0.1):
    from transformers import Qwen3_5TextConfig
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * n_blocks
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=64, intermediate_size=128,
        num_hidden_layers=len(layer_types), layer_types=layer_types, full_attention_interval=4,
        num_attention_heads=6, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6, attn_output_gate=True,
        max_position_embeddings=4096, tie_word_embeddings=False, initializer_range=init_range,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
        mtp_num_hidden_layers=1 if mtp else 0,
        bos_token_id=248044, eos_token_id=248044, pad_token_id=None,
    )
    cfg._attn_implementation = "eager"      # eager == the exact softmax path phase attention extends (bitwise tests)
    return cfg


def make_tiny_model(seed: int = 0, **kw):
    from transformers import Qwen3_5ForCausalLM
    torch.manual_seed(seed)
    cfg = tiny_config(**kw)
    m = Qwen3_5ForCausalLM(cfg).eval()
    # HF zero-inits the (1+w) RMSNorm weights and A_log/dt_bias deterministically; randomise the norms a bit
    # so the reference is not degenerate (all still deterministic under the seed).
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") and "linear_attn" not in n:
                p.add_(0.1 * torch.randn(p.shape, generator=g))
    return m
