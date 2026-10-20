#!/usr/bin/env python
"""Train ONLY the new layers on top of frozen Qwen3-0.6B (WikiText-2 TRAIN split only).
Loss = KL(stock teacher || student) + lm_w * CE + gate_w * mean gate-open rate.
Resumable checkpoints in ckpt/<arm>.pt; final trainable params in results/trained_params/<arm>.pt;
eval (untrained + trained) on the first 32,768 WikiText-2 TEST tokens -> results/trained.json."""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_arch import PhaseQwen3ForCausalLM, parse_layers
from qwen_arch.evaluate import wikitext_ids, perplexity, generations, paired_tests, probe, gate_layers

MID = "8-19"
HARD = {"type": "phase", "mode": "kv", "period": 4, "gain": 1.0, "supp": 0.0, "trainable": True}
SOFT = {"type": "phase", "mode": "kv", "period": 4, "gain": 2.0, "supp": 1.0, "trainable": True, "train_supp": True}
RESTORE = {"type": "suppressed_tracking", "period": 4, "coupling": "restore", "alpha": 0.7, "lam": 1.0,
           "gate": "either", "tau": 0.5, "trainable": True, "adapter_rank": 8}
TRACK = {"type": "tracking", "alpha": 0.9, "lam": 0.05, "gate": "harmony", "tau": 0.5, "trainable": True}
HR = [{"layers": MID, "attention": HARD}, {"layers": MID, "extra": RESTORE}]
ARMS = {
    "hard_restore_either": dict(rules=HR, gate_w=0.0),
    "hard_restore_either_gw0.05": dict(rules=HR, gate_w=0.05),
    "hard_restore_either_gw0.2": dict(rules=HR, gate_w=0.2),
    "hard_restore_either_gw1.0": dict(rules=HR, gate_w=1.0),
    "hard_alone": dict(rules=[{"layers": MID, "attention": {**HARD, "learn_phase": True, "phase_sharp": 20.0}}], gate_w=0.0),
    "soft_track": dict(rules=[{"layers": MID, "attention": SOFT}, {"layers": "every4@8-19", "extra": TRACK}], gate_w=0.0),
    "chunk4": dict(rules=[{"layers": "all", "position": {"type": "chunk_rope", "chunk": 4, "trainable": True, "blend": 0.99}}], gate_w=0.0),
    "CTRL_hard_restore_either_shuffled": dict(rules=[{"layers": MID, "attention": HARD},
                                                     {"layers": MID, "extra": {**RESTORE, "source": "shuffled"}}], gate_w=0.0),
    # domain-adaptation control: same trainable knobs, but phase starts as the identity (gain=supp=1, no suppression)
    "CTRL_identity_trainable": dict(rules=[{"layers": MID, "attention": {**HARD, "supp": 1.0, "train_supp": True}},
                                           {"layers": MID, "extra": RESTORE}], gate_w=0.0),
}
NEW_KEYS = ("log_gain", "log_out", "log_supp", "phase_offset", "log_period", "log_psharp", "lam_p", "mix_p",
            "alpha_logit", "tau_p", "log_sharp", "adapter", "blend_logit")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--lr_adapter", type=float, default=1e-3)
ap.add_argument("--lm_w", type=float, default=0.2)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--eval_windows", type=int, default=64)
ap.add_argument("--ckpt_every", type=int, default=25)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--out", default="results/trained.json")
args = ap.parse_args()
torch.set_num_threads(args.threads)
cfg = ARMS[args.arm]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
teacher = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
for p in teacher.parameters():
    p.requires_grad_(False)
student = PhaseQwen3ForCausalLM(teacher.config, parse_layers(teacher.config.num_hidden_layers, cfg["rules"]))
tparams = dict(teacher.named_parameters())
for n, p in student.named_parameters():          # share frozen base weights with the teacher (no copy)
    if n in tparams:
        p.data = tparams[n].data
student.lm_head.weight = student.embed_tokens.weight
is_new = lambda n: any(k in n for k in NEW_KEYS)
trainable = [(n, p) for n, p in student.named_parameters() if is_new(n)]
for n, p in student.named_parameters():
    p.requires_grad_(is_new(n))
n_train = sum(p.numel() for _, p in trainable)
print(f"[{args.arm}] trainable params: {n_train:,} in {len(trainable)} tensors", flush=True)
student.eval()


def save_json(key, val):
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault(args.arm, {})[key] = val
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out + ".tmp", "w"), indent=1); os.replace(args.out + ".tmp", args.out)


def load_json():
    return (json.load(open(args.out)) if os.path.exists(args.out) else {}).get(args.arm, {})


TEST = wikitext_ids(tok, "test")
WIN = [TEST[w * 512:(w + 1) * 512][None] for w in range(args.eval_windows)]


def evaluate(tag):
    r = perplexity(student, WIN)
    r["generations"], r["decode_tps"] = generations(student, tok)
    r["values"] = {n: (p.detach().flatten()[:16].tolist() if p.numel() > 1 else float(p)) for n, p in trainable if "adapter" not in n}
    print(f"[{args.arm}] {tag}: ppl {r['ppl']:.3f} gate {r.get('gate_mean')} open {r.get('gate_open_frac')} "
          f"prefill {r['prefill_tps']:.0f} decode {r['decode_tps']:.1f} | {r['generations'][0][:90]!r}", flush=True)
    return r


state = load_json()
if "untrained" not in state:
    save_json("untrained", evaluate("untrained"))
save_json("n_trainable", n_train)

os.makedirs("ckpt", exist_ok=True)
ck = f"ckpt/{args.arm}.pt"
groups = [{"params": [p for n, p in trainable if "adapter" not in n], "lr": args.lr},
          {"params": [p for n, p in trainable if "adapter" in n], "lr": args.lr_adapter}]
opt = torch.optim.Adam([g for g in groups if g["params"]])
step0 = 0
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False)
    student.load_state_dict(s["trainable"], strict=False); opt.load_state_dict(s["opt"]); step0 = s["step"]
    print(f"resumed at step {step0}", flush=True)
state = load_json()
if "trained" not in state and step0 < args.steps:
    TRAIN = wikitext_ids(tok, "train")       # TRAIN split only
    g = torch.Generator().manual_seed(1234)
    for _ in range(step0 * args.accum):      # deterministic data stream on resume
        torch.randint(0, TRAIN.numel() - args.seq - 1, (1,), generator=g)
    t0 = time.time(); log = open(f"logs/train_{args.arm}.jsonl", "a")
    for step in range(step0, args.steps):
        opt.zero_grad(set_to_none=True)
        tot = {"kl": 0.0, "ce": 0.0, "gate": 0.0}
        for _ in range(args.accum):
            i = int(torch.randint(0, TRAIN.numel() - args.seq - 1, (1,), generator=g))
            x = TRAIN[i:i + args.seq][None]
            with torch.no_grad():
                tl = F.log_softmax(teacher(x)[0], -1)
            sl = F.log_softmax(student(x)[0], -1)
            kl = (tl.exp() * (tl - sl)).sum(-1).mean()
            ce = F.nll_loss(sl[:-1], x[0, 1:])
            gl = [e.last_gate[:, 1:].mean() for e in gate_layers(student) if e.last_gate is not None and e.trainable]
            gate = torch.stack(gl).mean() if gl else torch.zeros(())
            ((kl + args.lm_w * ce + cfg["gate_w"] * gate) / args.accum).backward()
            tot["kl"] += float(kl) / args.accum; tot["ce"] += float(ce) / args.accum; tot["gate"] += float(gate) / args.accum
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        rec = {"step": step + 1, **{k: round(v, 5) for k, v in tot.items()}, "s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % 5 == 0:
            print(rec, flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save({"step": step + 1, "opt": opt.state_dict(),
                        "trainable": {n: p.detach().clone() for n, p in trainable}}, ck)
    os.makedirs("results/trained_params", exist_ok=True)
    torch.save({n: p.detach().clone() for n, p in trainable}, f"results/trained_params/{args.arm}.pt")

state = load_json()
if "trained" not in state:
    if os.path.exists(f"results/trained_params/{args.arm}.pt"):
        student.load_state_dict(torch.load(f"results/trained_params/{args.arm}.pt"), strict=False)
    r = evaluate("trained")
    r["steps"] = args.steps; r["train_tokens"] = args.steps * args.accum * args.seq
    save_json("trained", r)
    state = load_json()
stock = json.load(open("results/surgery.json"))["stock@64x512"]["window_nll"][: args.eval_windows]
tr, un = state["trained"]["window_nll"], state["untrained"]["window_nll"]
save_json("stats", {"trained_vs_untrained": paired_tests(tr, un), "trained_vs_stock": paired_tests(tr, stock),
                    "untrained_vs_stock": paired_tests(un, stock)})
if args.probe:
    save_json("probe_layer14", probe(student, WIN[:8], 14))
print("DONE", args.arm, flush=True)
