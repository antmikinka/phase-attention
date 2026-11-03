"""Tiny Qwen3.8-structure model: PyTorch (HF + qwen38 wrappers) vs llama.cpp qwen35/qwen35_phase logits.
Writes results/qwen38_gguf_check.json. Needs the llama.cpp-phase build (phase_logits tool) and the converter."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen38 import make_tiny_model, PhaseSpec, PhaseQwen38, TOKENIZER_DIR
from qwen38.export import save_stock, save_phase

LCPP = Path(os.environ.get("LLAMACPP_DIR", "/workspace/llama.cpp-phase"))
WORK = Path(os.environ.get("QWEN38_WORK", "/workspace/qwen38-tiny"))
PY = sys.executable
TOOL = LCPP / "build-cpu/bin/phase_logits"


def tokens(n=64):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    text = (ROOT / "data/tinystories_valid.txt").read_text()[:4000]
    ids = tok(text)["input_ids"][:n]
    assert len(ids) == n
    return ids


def convert(d: Path) -> Path:
    out = d / "model-f32.gguf"
    if not out.exists():
        r = subprocess.run([PY, str(LCPP / "convert_hf_to_gguf.py"), str(d), "--outtype", "f32", "--no-mtp", "--outfile", str(out)],
                           capture_output=True, text=True, cwd=str(LCPP))
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-6000:])
            raise SystemExit(f"convert failed for {d}")
    return out


def lcpp_logits(gguf: Path, ids, chunk, fa, kv="f16"):
    tf, of = gguf.parent / "tokens.txt", gguf.parent / f"logits_c{chunk}_fa{fa}_{kv}.f32"
    tf.write_text("\n".join(map(str, ids)))
    r = subprocess.run([str(TOOL), str(gguf), str(tf), str(of), str(chunk), str(fa), "4", kv], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-6000:])
        raise SystemExit(f"phase_logits failed for {gguf}")
    return np.fromfile(of, dtype=np.float32).reshape(len(ids), -1)


VARIANTS = {
    "kv_soft_strength0": PhaseSpec(mode="kv", supp=1.0),
    "kv_soft": PhaseSpec(mode="kv", supp=0.05, period=4.0, sharp=3.0, offset=0.3),
    "kv_hard": PhaseSpec(mode="kv", hard=True, period=4.0),
    "head_soft": PhaseSpec(mode="head", supp=0.1, period=4.0, sharp=3.0, offset=0.3),
    "kv_soft_delta_write": PhaseSpec(mode="kv", supp=0.05, period=4.0, sharp=3.0, offset=0.3,
                                     delta_write={"enabled": True, "supp": 0.3}),
}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ids = tokens(64)
    x = torch.tensor([ids])
    res = {"n_tokens": len(ids), "variants": {}}
    m = make_tiny_model(0)
    with torch.no_grad():
        ref = m(input_ids=x, use_cache=False).logits[0].float().numpy()
    save_stock(m, WORK / "stock")
    g = convert(WORK / "stock")
    out = {"gguf_mb": round(g.stat().st_size / 2**20, 1), "checks": {}}
    for kv in ("f16", "f32"):
        for chunk in (64, 7, 1):
            for fa in (0, 1):
                ll = lcpp_logits(g, ids, chunk, fa, kv)
                out["checks"][f"kv{kv}_chunk{chunk}_fa{fa}"] = float(np.abs(ll - ref).max())
    out["ref_logit_absmax"] = float(np.abs(ref).max())
    res["stock"] = out
    print("stock", json.dumps(out), flush=True)

    for name, spec in VARIANTS.items():
        m = make_tiny_model(0)
        pm = PhaseQwen38(m, spec)
        with torch.no_grad():
            ph = pm(x)[0].float().numpy()
        d = WORK / name
        save_phase(pm, d)
        g = convert(d)
        o = {"torch_vs_stock_maxdiff": float(np.abs(ph - ref).max()), "checks": {}}
        for kv in ("f16", "f32"):
          for chunk in (64, 7, 1):
            for fa in (0, 1):
                ll = lcpp_logits(g, ids, chunk, fa, kv)
                o["checks"][f"kv{kv}_chunk{chunk}_fa{fa}"] = float(np.abs(ll - ph).max())
                if chunk == 64 and fa == 0 and kv == "f32":
                    o["lcpp_vs_stock_maxdiff"] = float(np.abs(ll - ref).max())
                    o["argmax_agree_torch"] = float((ll.argmax(-1) == ph.argmax(-1)).mean())
        res["variants"][name] = o
        print(name, json.dumps(o), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/qwen38_gguf_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
