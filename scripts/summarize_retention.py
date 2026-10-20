"""results/retention.json -> markdown (stdout): table, paired tests vs controls, learned phase params."""
import json, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_arch.evaluate import paired_tests
d = json.load(open("results/retention.json"))
S = json.load(open("results/surgery.json"))["stock@64x512"]; stock = math.exp(sum(S["window_nll"]) / len(S["window_nll"]))
LAGS = ["4", "12", "28", "60"]; PL = ["1", "4", "8", "12", "28", "32", "60"]
E = {a: r["eval"] for a, r in d.items() if "eval" in r}
pc = lambda x: f"{100*x:.1f}"
print(f"Stock ppl {stock:.2f}; ppl budget (+3%) {1.03*stock:.2f}\n")
print("| arm | new params | readout params | ppl | Δ stock | gate mean/open | trained readout top-1 % @ lag " + " / ".join(LAGS) +
      " | fresh probe % @ lag " + " / ".join(PL) + " |")
print("|---|---|---|---|---|---|---|---|")
for a, e in E.items():
    g = f"{e['gate_mean']:.2f}/{e['gate_open_frac']:.2f}" if e.get("gate_mean") is not None else "-"
    print(f"| {a} | {d[a]['n_trainable_new']:,} | {d[a]['n_readout']:,} | {e['ppl']:.2f} | {100*(e['ppl']/stock-1):+.1f}% | {g} | "
          + " / ".join(pc(e["readout_acc"][k]) for k in LAGS) + " | " + " / ".join(pc(e["fresh_probe_acc"][k]) for k in PL) + " |")
PAIRS = [(a, c) for a in E for c in E if a != c and (
    (a.startswith("hard_restore_aux") and c in (a.replace("hard_restore", "CTRL_identity"), a.replace("hard_restore", "CTRL_shuffled"), "STOCK_readout"))
    or (a == "hard_alone_aux0.1" and c in ("CTRL_identity_small_aux0.1", "STOCK_readout"))
    or (a == "hard_restore_detach" and c == "CTRL_identity_detach")
    or (a.startswith("hard_restore_aux") and c == "hard_restore_detach"))]
print("\n### Paired tests (64 test windows): retention accuracy, arm − control (positive = arm retains more)\n")
print("| arm vs control | metric | " + " | ".join(f"lag {k}" for k in PL) + " |")
print("|---|---|" + "---|" * len(PL))
for a, c in PAIRS:
    for met, key, lags in (("readout", "readout_acc_per_window", LAGS), ("fresh", "fresh_probe_acc_per_window", PL)):
        cells = []
        for k in PL:
            if k not in lags:
                cells.append(""); continue
            x, y = E[a][key][k], E[c][key][k]
            t = paired_tests([-v for v in x], [-v for v in y])   # 'a_lower_in' then counts windows where arm acc is higher
            cells.append(f"{100*(sum(x)-sum(y))/len(x):+.1f} ({t['a_lower_in']}/64, p={t['sign_p']:.2g}/{t['wilcoxon_p']:.2g})")
        print(f"| {a} vs {c} | {met} | " + " | ".join(cells) + " |")
print("\n### Learned phase / gate params (layers 8, 14, 19)\n")
for a, e in E.items():
    v = e["values"]
    ks = [k for k in v if any(f"layers.{L}." in k for L in (8, 14, 19)) and any(s in k for s in ("psharp", "period", "offset", "lam_p", "tau_p", "log_supp"))]
    if ks:
        print(f"- **{a}**: " + "; ".join(f"`{k}`={[round(x,2) for x in v[k][:4]] if isinstance(v[k], list) else round(v[k],3)}" for k in ks))
print("\n### Generations\n")
for a, e in E.items():
    print(f"- **{a}**: " + " / ".join(repr(g[:120]) for g in e["generations"]))
