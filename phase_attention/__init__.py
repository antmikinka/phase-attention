"""phase-attention: a tiny, CPU-trainable LM architecture with a swappable
attention interface and a phase-alternating multi-stream prototype."""
from .attention import Attention, CausalSelfAttention, RotaryEmbedding, ATTENTION_REGISTRY, register_attention
from .model import ModelConfig, TinyLM, PhaseLM, build_model
from .chunks import chunk_reverse, chunk_positions
