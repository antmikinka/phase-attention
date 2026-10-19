#!/usr/bin/env python
"""Zero-training architecture surgery on pretrained Qwen3-0.6B.
For each variant: WikiText-2 ppl (first N windows of 512 test tokens), prefill
and decode TPS, greedy generations, chunk-reversal robustness (chunk variants),
and a suppressed-stream decoding probe (phase variants). Resumable."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_arch import PhaseQwen3ForCausalLM, parse_layers
from phase_attention.chunks import chunk_reverse

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=int, default=16)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--only", default=None, help="comma list of variants")
ap.add_argument("--out", default="results/surgery.json")
ap.add_argument("--probe_windows", type=int, default=8)
ap.add_argument("--rev_windows", type=int, default=8)
ap.add_argument("--gen_tokens", type=int, default=40)
args = ap.parse_args()
torch.set_num_threads(args.threads)

MID = "8-19"
KV = lambda **k: {"type": "phase", "mode": "kv", "period": 4, **k}
HD = lambda **k: {"type": "phase", "mode": "head", "period": 4, **k}
TR = lambda **k: {"type": "tracking", "alpha": 0.9, "lam": 0.05, **k}
VARIANTS = {
    "stock": [],
    "phase_kv_identity_all": [{"layers": "all", "attention": KV(gain=1.0, supp=1.0)}],
    "phase_kv_soft_mid": [{"layers": MID, "attention": KV(gain=2.0, supp=1.0)}],
    "phase_kv_hard_mid": [{"layers": MID, "attention": KV(gain=1.0, supp=0.0)}],
    "phase_kv_hard_mid_P16": [{"layers": MID, "attention": KV(gain=1.0, supp=0.0, period=16)}],
    "phase_kv_hard_all": [{"layers": "all", "attention": KV(gain=1.0, supp=0.0)}],
    "phase_head_hard_mid": [{"layers": MID, "attention": HD(gain=2.0, supp=0.0)}],
    "phase_head_soft_mid": [{"layers": MID, "attention": HD(gain=1.5, supp=0.5)}],
    "chunk2_all": [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 2}}],
    "chunk4_all": [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 4}}],
    "chunk8_all": [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 8}}],
    "chunk4_upper": [{"layers": "14-27", "position": {"type": "chunk_rope", "chunk": 4}}],
    "track_harmony_mid": [{"layers": "every4@8-19", "extra": TR(gate="harmony", tau=0.5)}],
    "track_always_mid": [{"layers": "every4@8-19", "extra": TR(gate="always")}],
    "track_phase_mid": [{"layers": "every4@8-19", "extra": TR(gate="phase", period=4)}],
    "combo_kvsoft_track": [{"layers": MID, "attention": KV(gain=2.0, supp=1.0)},
                           {"layers": "every4@8-19", "extra": TR(gate="harmony", tau=0.5)}],
}
PROBE = {"phase_kv_soft_mid": 14, "phase_kv_hard_mid": 14, "phase_head_hard_mid": 14, "phase_kv_hard_mid_P16": 14}
REV = {"stock": 4, "chunk2_all": 2, "chunk4_all": 4, "chunk8_all": 8, "chunk4_upper": 4}

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
import pyarrow.parquet as pq
pqp = Path("/workspace/krystal-qwen/data/wikitext/test.parquet")
if not pqp.exists():
    pqp = Path("data/wikitext_test.parquet")
    os.system(f"curl -sSL -o {pqp} https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet")
IDS = tok("\n\n".join(pq.read_table(pqp).column("text").to_pylist()), return_tensors="pt").input_ids[0]
WIN = [IDS[w * args.seq:(w + 1) * args.seq][None] for w in range(args.windows)]
PROMPTS = ["Explain in two sentences why the sky is blue.",
           "What is 3/4 + 1/8? Answer with a fraction.",
           "Write the first line of a short story about a lighthouse keeper."]

base = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
BASE_SD = base.state_dict()
res = json.load(open(args.out)) if os.path.exists(args.out) else {}


def save():
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out + ".tmp", "w"), indent=1); os.replace(args.out + ".tmp", args.out)


def build(rules):
    m = PhaseQwen3ForCausalLM(base.config, parse_layers(base.config.num_hidden_layers, rules))
    m.load_state_dict(BASE_SD, strict=False)
    return m.eval()


@torch.no_grad()
def ppl(m):
    nll, cnt, per = 0.0, 0, []
    t0 = time.time()
    for x in WIN:
        lg = m(x)[0, :-1]
        l = float(F.cross_entropy(lg, x[0, 1:], reduction="sum"))
        nll += l; cnt += x.shape[1] - 1; per.append(round(l / (x.shape[1] - 1), 5))
    dt = time.time() - t0
    return math.exp(nll / cnt), per, len(WIN) * args.seq / dt


@torch.no_grad()
def reversal(m, C):
    out = {}
    for name, f in (("intact", lambda x: x), ("reversed", lambda x: chunk_reverse(x, C))):
        nll, cnt = 0.0, 0
        for x in WIN[: args.rev_windows]:
            lg = m(f(x))[0, :-1]
            tgt = x[0, 1:]
            t = torch.arange(lg.shape[0])
            mask = ((t % C) == C - 1) & (t >= 2 * C)
            nll += float(F.cross_entropy(lg[mask], tgt[mask], reduction="sum")); cnt += int(mask.sum())
        out[name] = math.exp(nll / cnt)
    out["ratio"] = out["reversed"] / out["intact"]; out["chunk"] = C
    return out


def ridge_acc(Xtr, ytr, Xte, yte, ncls, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = F.one_hot(ytr, ncls).float()
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * Xtr.shape[0] * torch.eye(Xtr.shape[1]), Xtr.T @ (Y - Y.mean(0)))
    return float(((Xte @ W).argmax(-1) == yte).float().mean())


def last_positions(T, P):
    """For each query t: most recent s<t with phase != phase(t) (suppressed-stream key)
    and most recent s<t with phase == phase(t) (access-stream key)."""
    ph = (torch.arange(T) // P) % 2
    other = torch.full((T,), -1, dtype=torch.long); same = torch.full((T,), -1, dtype=torch.long)
    last = {0: -1, 1: -1}
    for t in range(T):
        p = int(ph[t])
        other[t] = last[1 - p]; same[t] = last[p]
        last[p] = t
    return other, same


@torch.no_grad()
def probe_v2(m, layer):
    att = m.layers[layer].self_attn
    att.record = True
    F_s, F_a, Y_o, Y_s = [], [], [], []
    for x in WIN[: args.probe_windows]:
        m(x)
        T = x.shape[1]
        other, same = last_positions(T, att.period)
        ok = (other >= 0) & (same >= 0)
        F_s.append(att.buffer["suppressed"][0][ok]); F_a.append(att.buffer["access_stream"][0][ok])
        Y_o.append(x[0][other[ok]]); Y_s.append(x[0][same[ok]])
    att.record = False
    F_s, F_a, Y_o, Y_s = map(torch.cat, (F_s, F_a, Y_o, Y_s))
    out = {"layer": layer, "n_rows": int(F_s.shape[0])}
    nwin = args.probe_windows
    for tname, Y in (("token_at_last_suppressed_pos", Y_o), ("token_at_last_access_pos", Y_s)):
        vals, cnts = Y.unique(return_counts=True)
        top = vals[cnts.argsort(descending=True)[:40]]
        keep = torch.isin(Y, top)
        remap = {int(v): i for i, v in enumerate(top)}
        y = torch.tensor([remap[int(v)] for v in Y[keep]])
        n = y.shape[0]; cut = int(n * 0.6)
        maj = float((y[cut:] == y[:cut].mode().values).float().mean())
        out[tname] = {"n": n, "majority_baseline": maj}
        for fname, Fm in (("from_suppressed_buffer", F_s), ("from_access_stream", F_a)):
            X = Fm[keep]
            out[tname][fname] = ridge_acc(X[:cut], y[:cut], X[cut:], y[cut:], 40)
    return out


@torch.no_grad()
def gen(m):
    outs, toks, secs = [], 0, 0.0
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"]
        t0 = time.time()
        o = m.generate(ids, args.gen_tokens, eos=tok.eos_token_id)
        secs += time.time() - t0; toks += o.shape[1]
        outs.append(tok.decode(o[0], skip_special_tokens=True))
    return outs, toks / secs


names = args.only.split(",") if args.only else list(VARIANTS)
for name in names:
    key = f"{name}@{args.windows}x{args.seq}"
    if key in res:
        continue
    m = build(VARIANTS[name])
    r = {"variant": name, "rules": VARIANTS[name], "windows": args.windows, "seq": args.seq,
         "threads": args.threads}
    if name == "stock" or name.endswith("identity_all"):
        from transformers import AutoModelForCausalLM
        x = WIN[0][:, :64]
        hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
        with torch.no_grad():
            a, b = m(x), hf(x).logits
        r["vs_hf"] = {"max_abs_diff": float((a - b).abs().max()), "bitwise_equal": bool(torch.equal(a, b)),
                      "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean())}
        del hf
    r["ppl"], r["window_nll"], r["prefill_tps"] = ppl(m)
    print(name, "ppl", round(r["ppl"], 3), "prefill tps", round(r["prefill_tps"], 1), flush=True)
    if name in REV and args.windows >= args.rev_windows:
        r["reversal"] = reversal(m, REV[name]); print("  reversal", r["reversal"], flush=True)
    if name in PROBE:
        r["probe"] = probe_v2(m, PROBE[name]); print("  probe", json.dumps(r["probe"]), flush=True)
    r["generations"], r["decode_tps"] = gen(m)
    print("  decode tps", round(r["decode_tps"], 2), "|", r["generations"][0][:120].replace("\n", " "), flush=True)
    res[key] = r
    save()
print("DONE", flush=True)
