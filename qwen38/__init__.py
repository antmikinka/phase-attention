"""Phase attention for the Qwen3.8-27B architecture (HF `qwen3_5` hybrid: 3x Gated DeltaNet + 1x gated full attention).

Only the *structure* of Qwen3.8-27B is used here; tests run on a tiny random-init copy (see tiny.py).
"""
from .tiny import tiny_config, make_tiny_model, TOKENIZER_DIR
from .phase import PhaseSpec, PhaseQwen38, PhaseQwenAttention, PhaseDecoderLayer, PhaseBetaProj
