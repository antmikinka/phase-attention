"""Shared evaluation helpers: WikiText-2 windows, perplexity, generations, probe, paired stats."""
from __future__ import annotations
import math, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

WT_TEST = Path("/workspace/krystal-qwen/data/wikitext/test.parquet")
WT_TRAIN = Path(__file__).resolve().parent.parent / "data" / "wikitext2_train.parquet"
PROMPTS = ["Explain in two sentences why the sky is blue.",
           "What is 3/4 + 1/8? Answer with a fraction.",
           "Write the first line of a short story about a lighthouse keeper."]


def wikitext_ids(tok, split="test"):
    import pyarrow.parquet as pq
    p = WT_TEST if split == "test" else WT_TRAIN
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        name = "test" if split == "test" else "train"
        os.system(f"curl -sSL -o {p} https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{name}-00000-of-00001.parquet")
    return tok("\n\n".join(pq.read_table(p).column("text").to_pylist()), return_tensors="pt").input_ids[0]


def gate_layers(m):
    return [e for l in m.layers for e in l.extras if hasattr(e, "last_gate")]


@torch.no_grad()
def perplexity(m, windows):
    nll, cnt, per, gates, opens = 0.0, 0, [], [], []
    t0 = time.time()
    for x in windows:
        lg = m(x)[0, :-1]
        l = float(F.cross_entropy(lg, x[0, 1:], reduction="sum"))
        nll += l; cnt += x.shape[1] - 1; per.append(round(l / (x.shape[1] - 1), 6))
        for e in gate_layers(m):
            if e.last_gate is not None:
                g = e.last_gate[:, 1:]
                gates.append(float(g.mean())); opens.append(float((g > 0.5).float().mean()))
    dt = time.time() - t0
    out = {"ppl": math.exp(nll / cnt), "window_nll": per, "prefill_tps": len(windows) * windows[0].shape[1] / dt}
    if gates:
        out["gate_mean"] = sum(gates) / len(gates); out["gate_open_frac"] = sum(opens) / len(opens)
    return out


@torch.no_grad()
def generations(m, tok, n_new=40):
    outs, toks, secs = [], 0, 0.0
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"]
        t0 = time.time()
        o = m.generate(ids, n_new, eos=tok.eos_token_id)
        secs += time.time() - t0; toks += o.shape[1]
        outs.append(tok.decode(o[0], skip_special_tokens=True))
    return outs, toks / secs


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def paired_tests(a, b):
    """a, b: per-window mean NLLs on the same windows. Exact two-sided sign test and
    Wilcoxon signed-rank (normal approximation, tie-corrected)."""
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    out = {"n": len(a), "mean_diff_nll": sum(x - y for x, y in zip(a, b)) / len(a),
           "a_lower_in": sum(1 for x, y in zip(a, b) if x < y)}
    if n == 0:
        out.update(sign_p=1.0, wilcoxon_p=1.0); return out
    k = sum(1 for x in d if x < 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    out["sign_p"] = min(1.0, 2 * tail)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    r = [0.0] * n; i = 0; ties = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for q in range(i, j + 1):
            r[order[q]] = (i + j) / 2 + 1
        t = j - i + 1; ties += t ** 3 - t; i = j + 1
    wp = sum(r[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - ties / 48)
    out["wilcoxon_p"] = min(1.0, 2 * _norm_sf((abs(wp - mu) - 0.5) / sd)) if sd > 0 else 1.0
    return out


def last_positions(T, P):
    ph = (torch.arange(T) // P) % 2
    other = torch.full((T,), -1, dtype=torch.long); same = torch.full((T,), -1, dtype=torch.long)
    last = {0: -1, 1: -1}
    for t in range(T):
        p = int(ph[t]); other[t] = last[1 - p]; same[t] = last[p]; last[p] = t
    return other, same


def ridge_acc(Xtr, ytr, Xte, yte, ncls, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = F.one_hot(ytr, ncls).float()
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * Xtr.shape[0] * torch.eye(Xtr.shape[1]), Xtr.T @ (Y - Y.mean(0)))
    return float(((Xte @ W).argmax(-1) == yte).float().mean())


@torch.no_grad()
def probe(m, windows, layer):
    att = m.layers[layer].self_attn
    att.record = True
    Fs, Fa, Yo, Ys = [], [], [], []
    for x in windows:
        m(x)
        other, same = last_positions(x.shape[1], att.period)
        ok = (other >= 0) & (same >= 0)
        Fs.append(att.buffer["suppressed"][0][ok]); Fa.append(att.buffer["access_stream"][0][ok])
        Yo.append(x[0][other[ok]]); Ys.append(x[0][same[ok]])
    att.record = False
    Fs, Fa, Yo, Ys = map(torch.cat, (Fs, Fa, Yo, Ys))
    out = {"layer": layer}
    for tname, Y in (("token_at_last_suppressed_pos", Yo), ("token_at_last_access_pos", Ys)):
        vals, cnts = Y.unique(return_counts=True)
        top = vals[cnts.argsort(descending=True)[:40]]
        keep = torch.isin(Y, top)
        remap = {int(v): i for i, v in enumerate(top)}
        y = torch.tensor([remap[int(v)] for v in Y[keep]])
        cut = int(len(y) * 0.6)
        out[tname] = {"n": len(y), "majority_baseline": float((y[cut:] == y[:cut].mode().values).float().mean()),
                      "from_suppressed_buffer": ridge_acc(Fs[keep][:cut], y[:cut], Fs[keep][cut:], y[cut:], 40),
                      "from_access_stream": ridge_acc(Fa[keep][:cut], y[:cut], Fa[keep][cut:], y[cut:], 40)}
    return out
