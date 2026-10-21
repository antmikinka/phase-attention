"""phase_model: a decoder-only LM built from scratch (no pretrained weights, own byte tokenizer)
whose blocks can each use a different mixer / position / extra layer. See docs/MODEL_FROM_SCRATCH.md."""
from .config import ModelConfig, BlockSpec, layer_specs
from .model import PhaseLM
from .tokenizer import ByteTokenizer
