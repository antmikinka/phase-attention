#!/usr/bin/env python
"""Train/eval one arm. Appends results to a JSON file after each eval.

Examples:
  python scripts/train.py --task rivalry --arm phase_fixed --steps 1500
  python scripts/train.py --task lm --arm baseline --steps 1500
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_attention import ModelConfig, build_model, chunk_reverse
from phase_attention.chunks import chunk_end_mask
from phase_attention.data import RivalryConfig, make_rivalry_batch, ByteText
from phase_attention.probe import ridge_probe

ARMS = {
    "baseline":            dict(arch="baseline"),
    "baseline_2x":         dict(arch="baseline", depth_mult=2),      # FLOPs-matched to K=2 streams
    "phase_fixed":         dict(arch="phase", schedule="fixed"),
    "phase_all":           dict(arch="phase", schedule="all"),        # no-phase control
    "phase_random":        dict(arch="phase", schedule="random"),     # random-phase control
    "phase_fixed_noinhib": dict(arch="phase", schedule="fixed", inhibition=False),
    "phase_learned":       dict(arch="phase", schedule="learned"),
    "coarse4":             dict(arch="baseline", rope_chunk=4),       # CoarseRoPE(C=4)
    "phase_mask":          dict(arch="baseline", attention="phase_mask"),
}


def make_cfg(args, vocab, channels):
    spec = dict(ARMS[args.arm])
    dm = spec.pop("depth_mult", 1)
    cfg = ModelConfig(vocab_size=vocab, n_channels=channels, d_model=args.d_model,
                      n_layers=args.n_layers * dm, n_heads=args.n_heads, period=args.period)
    for k, v in spec.items():
        setattr(cfg, k, v)
    if cfg.attention == "phase_mask":
        cfg.attention_kwargs = {"period": args.period, "n_phases": 2}
    return cfg


def lr_at(step, total, base, warm=100):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_rivalry(model, rc, seed=999, n=512):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids, tgt, lab = make_rivalry_batch(rc, n, g)
    logits, hid = model(ids, return_hidden=True)
    T, D = rc.seq_len, rc.delay
    nn_ = T - 2
    pred = logits.argmax(-1)
    rep_pos = torch.arange(2, nn_)
    dom_acc = (pred[:, rep_pos] == tgt[:, rep_pos]).float().mean().item()
    recall_acc = (pred[:, -1] == tgt[:, -1]).float().mean().item()
    # probes: suppressed eye's NEXT clean symbol (clean region) and its hidden
    # continuation during the noise (delay) region.
    half = n // 2
    def probe(feat, positions, labels_fn):
        X = feat[:, positions].reshape(-1, feat.shape[-1])
        y = labels_fn(positions).reshape(-1)
        Np = len(positions)
        return ridge_probe(X[: half * Np], y[: half * Np], X[half * Np:], y[half * Np:], rc.n_symbols)
    clean = torch.arange(3, nn_ - D - 1)
    delay = torch.arange(nn_ - D, nn_)
    sup_next = lambda p: lab["sup"][:, p]          # position p predicts index p (next symbol)
    dom_next = lambda p: lab["dom"][:, p]
    sup_now = lambda p: lab["sup"][:, p - 1]       # ground-truth continuation during noise
    feats = {"final": hid["final"]}
    if "streams" in hid:
        for k in range(hid["streams"].shape[1]):
            feats[f"stream{k}"] = hid["streams"][:, k]
    out = {"dom_acc": dom_acc, "recall_acc": recall_acc}
    for name, f in feats.items():
        out[f"probe_sup_next_{name}"] = probe(f, clean, sup_next)
        out[f"probe_sup_delay_{name}"] = probe(f, delay, sup_now)
        out[f"probe_dom_next_{name}"] = probe(f, clean, dom_next)
    model.train()
    return out


@torch.no_grad()
def eval_lm(model, data, T, n=64):
    model.eval()
    x, y = data.fixed_val(n, T)
    out = {}
    logits = model(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(n, T)
    out["val_bpb"] = (nll.mean() / math.log(2)).item()
    for C in (2, 4, 8):
        m = chunk_end_mask(T, C)
        m[: C * 2] = False   # skip first chunks
        lr = model(chunk_reverse(x, C))
        nll_r = F.cross_entropy(lr.reshape(-1, lr.shape[-1]), y.reshape(-1), reduction="none").view(n, T)
        a, b = nll[:, m].mean() / math.log(2), nll_r[:, m].mean() / math.log(2)
        out[f"rev{C}_intact_bpb"] = a.item()
        out[f"rev{C}_reversed_bpb"] = b.item()
        out[f"rev{C}_ratio"] = (b / a).item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rivalry", "lm"], required=True)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--period", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--recall", default="last", choices=["last", "extrap"])
    ap.add_argument("--delay", type=int, default=8)
    ap.add_argument("--data", default="data/tinystories_valid.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.task == "rivalry":
        rc = RivalryConfig(seq_len=args.seq, delay=args.delay, recall=args.recall)
        cfg = make_cfg(args, rc.vocab_size, 2)
        g = torch.Generator().manual_seed(args.seed)
    else:
        data = ByteText(args.data)
        cfg = make_cfg(args, 256, 1)
        g = np.random.default_rng(args.seed)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    out_path = args.out or f"results/smoke_{args.task}.json"
    rec = {"task": args.task + ("_" + args.recall if args.task == "rivalry" else ""), "arm": args.arm, "seed": args.seed, "config": cfg.to_dict(),
           "args": vars(args), "n_params": model.n_params(), "evals": []}
    print(f"[{args.task}/{args.arm}] params={model.n_params():,}", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, args.steps, args.lr)
        if args.task == "rivalry":
            ids, tgt, _ = make_rivalry_batch(rc, args.batch, g)
            logits = model(ids)
            rep = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt[:, :-1].reshape(-1), ignore_index=-100)
            rec_loss = F.cross_entropy(logits[:, -1], tgt[:, -1])
            loss = rep + rec_loss
        else:
            x, y = data.batch("train", args.batch, args.seq, g)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = eval_rivalry(model, rc) if args.task == "rivalry" else eval_lm(model, data, args.seq)
            ev.update(step=step + 1, train_loss=loss.item(), elapsed_s=time.time() - t0)
            rec["evals"].append(ev)
            print("EVAL", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
            _save(out_path, rec)
    if isinstance(model, torch.nn.Module) and args.arm.startswith("phase") and hasattr(model, "inhib"):
        rec["learned"] = {"inhib": model.inhib.tolist(), "feedback": model.feedback.tolist(),
                          "access_gain_mean": model.access_gain.mean().item()}
        _save(out_path, rec)


def _save(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[f"{rec['arm']}_s{rec['seed']}"] = rec
    tmp = path + ".tmp"
    json.dump(allr, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
