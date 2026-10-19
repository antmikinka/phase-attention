"""Own Qwen3 model definition with per-layer pluggable layer types
(PhaseAttention, ChunkRoPE, Tracking) that loads pretrained Qwen3 weights."""
from .registry import ATTENTION_TYPES, POSITION_TYPES, EXTRA_LAYER_TYPES, register
from .modeling import Qwen3Config, PhaseQwen3ForCausalLM, LayerSpec, parse_layers
from . import layers  # noqa: F401  (registers built-in types)
