#!/usr/bin/env python
"""Retention reward: train ONLY the new layers + a small linear readout on frozen Qwen3-0.6B (WikiText-2 TRAIN only).

Loss = KL(stock || student) + lm_w*CE + aux_w * mean_k CE(readout_k(h19[t]) -> token[t-k]),  k in LAGS.
LAGS = 4, 12, 28, 60 are an odd number of P=4 blocks back, so token[t-k] is always in the *suppressed* phase
relative to the query t (hard-phase layers 8-19 cannot attend to it directly).
h19 = residual stream after layer 19 (end of the modified block), parameter-free RMS-normalised.
Readout_k = Linear(D,64) -> Linear(64, 2049) (+bias) over the 2048 most frequent TRAIN tokens + 'other'; identical for every arm.
Accuracy = argmax over the 2048 real tokens (never 'other'); 'other' targets count as wrong.
Eval on the first 32,768 WikiText-2 TEST tokens: ppl (64x512), trained-readout top-1 retention per lag (positions
64,71,...,505 of each window), and a FRESH ridge probe fit on TRAIN windows only (top-200 target classes) and scored
on the test windows, for lags 1,4,8,12,28,32,60 (8,32 = same-phase contrast). -> results/retention.json"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_arch import PhaseQwen3ForCausalLM, parse_layers
from qwen_arch.evaluate import wikitext_ids, perplexity, generations, paired_tests, gate_layers

MID = "8-19"; READ_LAYER = 19
LAGS = [4, 12, 28, 60]; PROBE_LAGS = [1, 4, 8, 12, 28, 32, 60]
HARD = {"type": "phase", "mode": "kv", "period": 4, "gain": 1.0, "supp": 0.0, "trainable": True}
IDENT = {**HARD, "supp": 1.0, "train_supp": True}
RESTORE = {"type": "suppressed_tracking", "period": 4, "coupling": "restore", "alpha": 0.7, "lam": 1.0,
           "gate": "either", "tau": 0.5, "trainable": True, "adapter_rank": 8}
HR = [{"layers": MID, "attention": HARD}, {"layers": MID, "extra": RESTORE}]
ID = [{"layers": MID, "attention": IDENT}, {"layers": MID, "extra": RESTORE}]
SH = [{"layers": MID, "attention": HARD}, {"layers": MID, "extra": {**RESTORE, "source": "shuffled"}}]
ARMS = {
    "hard_restore_aux0.1": dict(rules=HR, aux_w=0.1),
    "CTRL_identity_aux0.1": dict(rules=ID, aux_w=0.1),
    "CTRL_shuffled_aux0.1": dict(rules=SH, aux_w=0.1),
    "STOCK_readout": dict(rules=[], aux_w=0.1),                      # no new layers; readout only
    "hard_restore_aux0.3": dict(rules=HR, aux_w=0.3),
    "CTRL_identity_aux0.3": dict(rules=ID, aux_w=0.3),
    "hard_restore_aux0.03": dict(rules=HR, aux_w=0.03),
    "CTRL_identity_aux0.03": dict(rules=ID, aux_w=0.03),
    "hard_alone_aux0.1": dict(rules=[{"layers": MID, "attention": {**HARD, "learn_phase": True, "phase_sharp": 20.0}}], aux_w=0.1),
    "CTRL_identity_small_aux0.1": dict(rules=[{"layers": MID, "attention": IDENT}], aux_w=0.1),
    "hard_restore_detach": dict(rules=HR, aux_w=0.1, detach=True),   # readout trained, no reward reaches new layers
    "CTRL_identity_detach": dict(rules=ID, aux_w=0.1, detach=True),
}
NEW_KEYS = ("log_gain", "log_out", "log_supp", "phase_offset", "log_period", "log_psharp", "lam_p", "mix_p",
            "alpha_logit", "tau_p", "log_sharp", "adapter", "blend_logit")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--lr_adapter", type=float, default=1e-3)
ap.add_argument("--lr_readout", type=float, default=5e-3)
ap.add_argument("--lm_w", type=float, default=0.2)
ap.add_argument("--vocab", type=int, default=2048)
ap.add_argument("--n_pos", type=int, default=48)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--eval_windows", type=int, default=64)
ap.add_argument("--probe_train_windows", type=int, default=24)
ap.add_argument("--ckpt_every", type=int, default=25)
ap.add_argument("--reeval", action="store_true", help="drop the stored eval and recompute it")
ap.add_argument("--out", default="results/retention.json")
args = ap.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(0)
cfg = ARMS[args.arm]


class Readout(nn.Module):
    """Per lag: Linear(D,64) -> Linear(64, V+1) with bias, over the V most frequent TRAIN-split tokens + 'other'."""
    def __init__(self, D, vocab, lags, r=64):
        super().__init__()
        self.lags = lags; V = len(vocab)
        self.register_buffer("lut", torch.full((152000,), V, dtype=torch.long))
        self.lut[vocab] = torch.arange(V)
        self.down = nn.ModuleList(nn.Linear(D, r, bias=False) for _ in lags)
        self.up = nn.ModuleList(nn.Linear(r, V + 1) for _ in lags)
        self.V = V

    def forward(self, i, h):
        return self.up[i](self.down[i](h))

    def target(self, ids):
        return self.lut[ids]


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
teacher = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
for p in teacher.parameters():
    p.requires_grad_(False)
student = PhaseQwen3ForCausalLM(teacher.config, parse_layers(teacher.config.num_hidden_layers, cfg["rules"]))
tparams = dict(teacher.named_parameters())
for n, p in student.named_parameters():
    if n in tparams:
        p.data = tparams[n].data
student.lm_head.weight = student.embed_tokens.weight
is_new = lambda n: any(k in n for k in NEW_KEYS)
for n, p in student.named_parameters():
    p.requires_grad_(is_new(n))
trainable = [(n, p) for n, p in student.named_parameters() if is_new(n)]
TRAIN = wikitext_ids(tok, "train")         # TRAIN split only
_v, _c = TRAIN.unique(return_counts=True)
readout = Readout(teacher.config.hidden_size, _v[_c.argsort(descending=True)[:args.vocab]], LAGS)
n_new = sum(p.numel() for _, p in trainable); n_read = sum(p.numel() for p in readout.parameters())
print(f"[{args.arm}] trainable new-layer params {n_new:,}; readout params {n_read:,}", flush=True)
student.eval()

H = {}
student.layers[READ_LAYER].register_forward_hook(lambda m, i, o: H.__setitem__("h", o))
rms = lambda h: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


def save_json(key, val):
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault(args.arm, {})[key] = val
    json.dump(res, open(args.out + ".tmp", "w"), indent=1); os.replace(args.out + ".tmp", args.out)


def load_json():
    return (json.load(open(args.out)) if os.path.exists(args.out) else {}).get(args.arm, {})


save_json("n_trainable_new", n_new); save_json("n_readout", n_read)
os.makedirs("ckpt", exist_ok=True)
ck = f"ckpt/ret_{args.arm}.pt"; final = f"results/retention_params/{args.arm}.pt"
groups = [{"params": [p for n, p in trainable if "adapter" not in n], "lr": args.lr},
          {"params": [p for n, p in trainable if "adapter" in n], "lr": args.lr_adapter},
          {"params": list(readout.parameters()), "lr": args.lr_readout}]
opt = torch.optim.Adam([g for g in groups if g["params"]])
step0 = 0
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False)
    student.load_state_dict(s["trainable"], strict=False); readout.load_state_dict(s["readout"]); opt.load_state_dict(s["opt"]); step0 = s["step"]
    print(f"resumed at step {step0}", flush=True)
if not os.path.exists(final):
    g = torch.Generator().manual_seed(1234); gp = torch.Generator().manual_seed(99)
    for _ in range(step0 * args.accum):
        torch.randint(0, TRAIN.numel() - args.seq - 1, (1,), generator=g)
        for k in LAGS:
            torch.randint(k, args.seq, (args.n_pos,), generator=gp)
    t0 = time.time(); log = open(f"logs/ret_{args.arm}.jsonl", "a")
    for step in range(step0, args.steps):
        opt.zero_grad(set_to_none=True)
        tot = {"kl": 0.0, "ce": 0.0, "aux": 0.0, "gate": 0.0}
        for _ in range(args.accum):
            i = int(torch.randint(0, TRAIN.numel() - args.seq - 1, (1,), generator=g))
            x = TRAIN[i:i + args.seq][None]
            with torch.no_grad():
                tl = F.log_softmax(teacher(x)[0], -1)
            sl = F.log_softmax(student(x)[0], -1)
            h = rms(H["h"][0]); h = h.detach() if cfg.get("detach") or not trainable else h
            kl = (tl.exp() * (tl - sl)).sum(-1).mean()
            ce = F.nll_loss(sl[:-1], x[0, 1:])
            aux = 0
            for j, k in enumerate(LAGS):
                t = torch.randint(k, args.seq, (args.n_pos,), generator=gp)
                aux = aux + F.cross_entropy(readout(j, h[t]), readout.target(x[0, t - k])) / len(LAGS)
            gl = [e.last_gate[:, 1:].mean() for e in gate_layers(student) if e.last_gate is not None]
            loss = (kl + args.lm_w * ce if trainable else 0) + cfg["aux_w"] * aux
            (loss / args.accum).backward()
            for kk, v in (("kl", kl), ("ce", ce), ("aux", aux), ("gate", torch.stack(gl).mean() if gl else torch.zeros(()))):
                tot[kk] += float(v) / args.accum
        if trainable:
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        torch.nn.utils.clip_grad_norm_(list(readout.parameters()), 1.0)   # clipped separately
        opt.step()
        rec = {"step": step + 1, **{k: round(v, 5) for k, v in tot.items()}, "s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % 5 == 0:
            print(rec, flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save({"step": step + 1, "opt": opt.state_dict(), "readout": readout.state_dict(),
                        "trainable": {n: p.detach().clone() for n, p in trainable}}, ck)
    os.makedirs("results/retention_params", exist_ok=True)
    torch.save({"trainable": {n: p.detach().clone() for n, p in trainable}, "readout": readout.state_dict()}, final)
s = torch.load(final, weights_only=False)
student.load_state_dict(s["trainable"], strict=False); readout.load_state_dict(s["readout"])

state = load_json()
if args.reeval:
    state.pop("eval", None)
TEST = wikitext_ids(tok, "test")
WIN = [TEST[w * 512:(w + 1) * 512][None] for w in range(args.eval_windows)]
POS = torch.arange(64, 512, 7)
if "eval" not in state:
    with torch.no_grad():
        r = perplexity(student, WIN)
        r["generations"], r["decode_tps"] = generations(student, tok)
        r["values"] = {n: (p.detach().flatten()[:16].tolist() if p.numel() > 1 else float(p)) for n, p in trainable if "adapter" not in n}
        # trained readout retention (test windows; readout never saw test tokens)
        acc = {k: [] for k in LAGS}; feats = []
        for x in WIN:
            student(x); h = rms(H["h"][0]); feats.append(h[POS])
            for j, k in enumerate(LAGS):
                tg = readout.target(x[0, POS - k]); acc[k].append(float(((readout(j, h[POS])[:, :readout.V].argmax(-1) == tg) & (tg < readout.V)).float().mean()))
        r["readout_acc_per_window"] = {str(k): v for k, v in acc.items()}
        r["readout_acc"] = {str(k): sum(v) / len(v) for k, v in acc.items()}
        # fresh probe: ridge fit on TRAIN windows only, scored on test windows
        gtr = torch.Generator().manual_seed(7)
        Xtr, Ytr = [], []
        for _ in range(args.probe_train_windows):
            i = int(torch.randint(0, TRAIN.numel() - 513, (1,), generator=gtr)); x = TRAIN[i:i + 512][None]
            student(x); Xtr.append(rms(H["h"][0])[POS]); Ytr.append(x[0])
        Xtr = torch.cat(Xtr); Xte = torch.cat(feats)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        A, B = (Xtr - mu) / sd, (Xte - mu) / sd
        G = A.T @ A + 0.1 * A.shape[0] * torch.eye(A.shape[1])
        fp, fpw = {}, {}
        for k in PROBE_LAGS:
            ytr = torch.cat([y[POS - k] for y in Ytr]); yte = torch.cat([x[0, POS - k] for x in WIN])
            vals, cnts = ytr.unique(return_counts=True); top = vals[cnts.argsort(descending=True)[:200]]
            lut = torch.full((int(max(ytr.max(), yte.max())) + 1,), -1, dtype=torch.long); lut[top] = torch.arange(len(top))
            ctr, cte = lut[ytr], lut[yte]
            Y = F.one_hot(ctr.clamp(min=0), 200).float() * (ctr >= 0)[:, None]
            W = torch.linalg.solve(G, A.T @ (Y - Y.mean(0)))
            ok = (B @ W + Y.mean(0)).argmax(-1) == cte   # ridge with intercept (class prior kept)          # out-of-top-200 targets (cte=-1) count as wrong
            fp[str(k)] = float(ok.float().mean())
            fpw[str(k)] = ok.view(len(WIN), -1).float().mean(1).tolist()
        r["fresh_probe_acc"] = fp; r["fresh_probe_acc_per_window"] = fpw
    print(f"[{args.arm}] ppl {r['ppl']:.3f} gate {r.get('gate_mean')} open {r.get('gate_open_frac')} readout {r['readout_acc']} fresh {fp}", flush=True)
    save_json("eval", r)
stock = json.load(open("results/surgery.json"))["stock@64x512"]["window_nll"][: args.eval_windows]
save_json("ppl_vs_stock", paired_tests(load_json()["eval"]["window_nll"], stock))
print("DONE", args.arm, flush=True)
