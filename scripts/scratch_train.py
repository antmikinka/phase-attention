#!/usr/bin/env python
"""Train a phase_model LM FROM SCRATCH (no pretrained weights, byte tokenizer) on one task, one arm, one seed.
task lm     : TinyStories bytes (train split only), val bpb on 64x256 windows of the valid split.
task recall : synthetic phase-aligned associative recall (fresh random sequences; eval set from a held-out seed).
Resumable checkpoints ckpt/scratch/<run>.pt; result results/scratch/<run>.json."""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_model import ModelConfig, PhaseLM, layer_specs
from phase_model.data import tinystories, lm_batch, val_windows, recall_batch, LAG_BLOCKS

PH = {"type": "phase", "mode": "kv", "period": 4, "sharp": 4.0, "supp": 0.05}
ARMS = {
    "baseline": [],
    "phase_kv_fb": [{"mixer": PH}, {"extra": {"type": "feedback", "gate": "either"}}],
    "phase_head": [{"mixer": {**PH, "mode": "head"}}],
    "phase_track": [{"mixer": PH}, {"layers": "every2", "extra": {"type": "tracking"}}],
    "chunkrope": [{"position": {"type": "chunk_rope", "chunk": 4, "blend": 0.5}}],
    "CTRL_shuffled_fb": [{"mixer": PH}, {"extra": {"type": "feedback", "gate": "either", "source": "shuffled"}}],
    "phase_kv": [{"mixer": PH}],
}
D, L, H, BASE_HIDDEN = 256, 8, 8, 688


def build(arm, seed):
    """Match total params to the baseline by adjusting the (shared) MLP hidden size."""
    torch.manual_seed(seed)
    target = PhaseLM(ModelConfig(d_model=D, n_layers=L, n_heads=H, mlp_hidden=BASE_HIDDEN)).n_params()
    specs = layer_specs(L, ARMS[arm])
    best = None
    for hid in range(BASE_HIDDEN - 60, BASE_HIDDEN + 1):
        c = ModelConfig(d_model=D, n_layers=L, n_heads=H, mlp_hidden=hid, blocks=specs)
        n = PhaseLM(c).n_params()
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (hid, n)
    torch.manual_seed(seed)
    c = ModelConfig(d_model=D, n_layers=L, n_heads=H, mlp_hidden=best[0], blocks=specs)
    return PhaseLM(c), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["lm", "recall"], required=True)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    steps = a.steps or (1500 if a.task == "lm" else 1000)
    B, T = (16, 256) if a.task == "lm" else (64, 128)
    run = f"{a.task}_{a.arm}_s{a.seed}"
    os.makedirs("ckpt/scratch", exist_ok=True); os.makedirs("results/scratch", exist_ok=True); os.makedirs("logs/scratch", exist_ok=True)
    out = f"results/scratch/{run}.json"
    if os.path.exists(out):
        print("exists", out); return
    model, target = build(a.arm, a.seed)
    n = model.n_params()
    print(f"[{run}] params {n:,} (baseline {target:,}, {100*(n/target-1):+.2f}%)", flush=True)
    decay = [p for nme, p in model.named_parameters() if p.dim() >= 2]
    other = [p for nme, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    warm = 100
    sched = lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ck = f"ckpt/scratch/{run}.pt"; step0 = 0
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step0 = s["step"]
        print(f"resumed at {step0}", flush=True)
    g = torch.Generator().manual_seed(1000 + a.seed)
    for _ in range(step0):                         # replay data RNG for exact resume
        if a.task == "lm":
            torch.randint(0, 10, (B,), generator=g)
    if a.task == "lm":
        TRAIN = tinystories("train")               # TRAIN split only
    log = open(f"logs/scratch/{run}.jsonl", "a"); t0 = time.time(); model.train()
    for step in range(step0, steps):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * sched(step)
        if a.task == "lm":
            x = lm_batch(TRAIN, B, T, g)
            loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        else:
            x, pos, lag = recall_batch(B, T, g=g)
            logits = model(x[:, :-1])
            bi = torch.arange(B)[:, None].expand_as(pos)
            loss = F.cross_entropy(logits[bi, pos].reshape(-1, 256), x[bi, pos + 1].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "loss": round(loss.item(), 4), "s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (step + 1) % 100 == 0:
                print(rec, flush=True)
        if (step + 1) % a.ckpt_every == 0 or step + 1 == steps:
            torch.save({"step": step + 1, "model": model.state_dict(), "opt": opt.state_dict()}, ck)
    # ---------------------------------------------------------------- eval
    model.eval(); res = {"run": run, "task": a.task, "arm": a.arm, "seed": a.seed, "params": n,
                         "baseline_params": target, "steps": steps, "batch": [B, T], "train_seconds": round(time.time() - t0, 1)}
    with torch.no_grad():
        if a.task == "lm":
            W = val_windows(tinystories("val"), 64, 256)
            losses = []
            t1 = time.time()
            for w in W:
                lo = model(w[None, :-1]); losses.append(float(F.cross_entropy(lo[0], w[1:])))
            res["val_tok_per_s"] = 64 * 256 / (time.time() - t1)
            res["window_loss"] = losses
            res["val_loss"] = sum(losses) / len(losses); res["val_bpb"] = res["val_loss"] / math.log(2)
            res["val_ppl_byte"] = math.exp(res["val_loss"])
        else:
            ge = torch.Generator().manual_seed(777)       # held-out eval sequences (never used in training)
            corr = {}; per_batch = []
            t1 = time.time(); ntok = 0
            for _ in range(32):
                x, pos, lag = recall_batch(64, 128, g=ge)
                lo = model(x[:, :-1]); ntok += 64 * 128
                bi = torch.arange(64)[:, None].expand_as(pos)
                ok = (lo[bi, pos].argmax(-1) == x[bi, pos + 1])
                pb = {}
                for Lb in LAG_BLOCKS:
                    m = lag == Lb * 4
                    corr.setdefault(Lb * 4, []).append(ok[m].float())
                    pb[str(Lb * 4)] = float(ok[m].float().mean()) if m.any() else None
                per_batch.append(pb)
            res["val_tok_per_s"] = ntok / (time.time() - t1)
            res["recall_acc"] = {str(k): float(torch.cat(v).mean()) for k, v in corr.items()}
            res["recall_acc_per_batch"] = per_batch
            res["recall_acc_cross_phase"] = float(torch.cat([torch.cat(corr[k]) for k in corr if (k // 4) % 2 == 1]).mean())
            res["recall_acc_same_phase"] = float(torch.cat([torch.cat(corr[k]) for k in corr if (k // 4) % 2 == 0]).mean())
        model(torch.randint(0, 256, (1, T)))                # refresh gate stats on one forward
        res["phase_report"] = model.phase_report()
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    print(f"[{run}] DONE", {k: res[k] for k in res if k in ("val_bpb", "recall_acc", "recall_acc_cross_phase", "recall_acc_same_phase", "val_tok_per_s")}, flush=True)


if __name__ == "__main__":
    main()
