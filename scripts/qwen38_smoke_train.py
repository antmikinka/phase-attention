"""Frozen-base training smoke run on the tiny Qwen3.8-structure model (random base weights, so the numbers only
show that the loop, gradients and gates work - not that phase attention helps).
loss = KL(stock || phase) + 0.2 * LM + gate_w * mean(gate open rate). Writes results/qwen38_smoke_train.json."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen38 import make_tiny_model, PhaseSpec, PhaseQwen38, TOKENIZER_DIR


def batches(T=64, B=2, n=40, seed=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    ids = tok((ROOT / "data/tinystories_valid.txt").read_text()[:200_000])["input_ids"]
    ids = torch.tensor(ids)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        st = torch.randint(0, len(ids) - T - 1, (B,), generator=g)
        yield torch.stack([ids[s:s + T + 1] for s in st])


def main(steps=30, lr=1e-2, gate_w=0.01):
    torch.manual_seed(0)
    spec = PhaseSpec(mode="kv", supp=0.9, period=4.0, sharp=3.0,
                     feedback={"rank": 8, "lam": 0.0, "gate": "either"}, tracking={"rank": 8, "lam": 0.0})
    pm = PhaseQwen38(make_tiny_model(0), spec)
    for p in pm.hf_base_parameters():
        p.requires_grad_(False)
    new = pm.new_parameters()
    opt = torch.optim.AdamW(new, lr=lr, weight_decay=0.0)
    base_before = [p.detach().clone() for p in pm.hf_base_parameters()]
    log, t0 = [], time.time()
    for step, xb in enumerate(batches(n=steps)):
        x, y = xb[:, :-1], xb[:, 1:]
        with torch.no_grad():
            pm.set_active(False)
            teacher = F.log_softmax(pm(x).float(), -1)
            pm.set_active(True)
        logits = pm(x).float()
        logp = F.log_softmax(logits, -1)
        kl = F.kl_div(logp, teacher, log_target=True, reduction="batchmean") / x.shape[1]
        lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        gates = pm.gate_values()
        gate = torch.stack([g.mean() for g in gates]).mean() if gates else torch.tensor(0.0)
        loss = kl + 0.2 * lm + gate_w * gate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in new if p.grad is not None))
        opt.step()
        a = pm.attn[3]
        rec = {"step": step, "loss": float(loss), "kl": float(kl), "lm": float(lm), "gate_open": float(gate),
               "grad_norm": float(gn), "supp_l3_mean": float(a.supp().mean()), **{f"clock_l3_{k}": v for k, v in a.clock.values().items()},
               "fb_lam_l3": float(pm.dec[3].feedback.lam), "trk_lam_l3": float(pm.dec[3].tracking.lam)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    unchanged = all(torch.equal(a, b) for a, b in zip(base_before, pm.hf_base_parameters()))
    out = {"steps": steps, "lr": lr, "gate_w": gate_w, "seconds": round(time.time() - t0, 1),
           "n_new_params": sum(p.numel() for p in new), "n_base_params": sum(p.numel() for p in pm.hf_base_parameters()),
           "base_unchanged": unchanged, "log": log}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/qwen38_smoke_train.json").write_text(json.dumps(out, indent=2))
    print("base_unchanged", unchanged, "new", out["n_new_params"], "base", out["n_base_params"])


if __name__ == "__main__":
    main()
