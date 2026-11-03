"""Export the tiny (or any) wrapped model as HF directories for llama.cpp's converter.

stock dir : plain Qwen3_5ForCausalLM (config.json + model.safetensors + tokenizer files)
phase dir : same base weights, config.json architectures=["Qwen3_5PhaseForCausalLM"] + "phase_config",
            plus phase.safetensors with the phase tensors (see llamacpp/convert notes in docs/QWEN38_PHASE.md)
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file

from .tiny import TOKENIZER_DIR

TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
_PHASE_KEYS = (".clock.", "log_supp", "log_wsupp", ".feedback.", ".tracking.")


def base_state_dict(hf_model):
    sd = {}
    for k, v in hf_model.state_dict().items():
        if any(p in k for p in _PHASE_KEYS):
            continue
        sd[k.replace(".orig.", ".")] = v.detach().contiguous().clone()
    return sd


def _write(out: Path, hf_model, arch: str, extra_cfg: dict | None = None):
    out.mkdir(parents=True, exist_ok=True)
    cfg = hf_model.config.to_dict()
    cfg["architectures"] = [arch]
    cfg.pop("_attn_implementation_autoset", None)
    if extra_cfg:
        cfg.update(extra_cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(base_state_dict(hf_model), str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in TOK_FILES:
        shutil.copy(Path(TOKENIZER_DIR) / f, out / f)


def save_stock(hf_model, out):
    _write(Path(out), hf_model, "Qwen3_5ForCausalLM")


def save_phase(pm, out):
    st, meta = pm.phase_state()
    out = Path(out)
    _write(out, pm.hf, "Qwen3_5PhaseForCausalLM", {"phase_config": meta})
    save_file({k: v.contiguous() for k, v in st.items()}, str(out / "phase.safetensors"))
