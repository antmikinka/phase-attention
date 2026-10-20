"""results/trained.json -> markdown table (stdout)."""
import json, math
d = json.load(open("results/trained.json")); S = json.load(open("results/surgery.json"))["stock@64x512"]
stock = math.exp(sum(S["window_nll"]) / len(S["window_nll"]))
f = lambda x: "-" if x is None else f"{x:.3g}"
print(f"Stock Qwen3-0.6B (same 64x512 windows): ppl {stock:.2f}\n")
print("| arm | trainable params | untrained ppl | trained ppl | Δ vs stock | gate mean / open (trained) | prefill / decode TPS | sign p, Wilcoxon p vs stock | vs untrained | mean Δnll vs stock (wins/64) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a, r in d.items():
    if "trained" not in r: continue
    u, t, s = r["untrained"], r["trained"], r.get("stats", {})
    vs, vu = s.get("trained_vs_stock", {}), s.get("trained_vs_untrained", {})
    g = f"{t['gate_mean']:.3f} / {t['gate_open_frac']:.3f}" if t.get("gate_mean") is not None else "-"
    print(f"| {a} | {r['n_trainable']:,} | {u['ppl']:.2f} | **{t['ppl']:.2f}** | {100*(t['ppl']/stock-1):+.1f}% | {g} | "
          f"{t['prefill_tps']:.0f} / {t['decode_tps']:.1f} | {f(vs.get('sign_p'))}, {f(vs.get('wilcoxon_p'))} | "
          f"{f(vu.get('sign_p'))}, {f(vu.get('wilcoxon_p'))} | {f(vs.get("mean_diff_nll"))} ({vs.get("a_lower_in","-")}) |")
print("\n### Learned scalar values (first entries)\n")
for a, r in d.items():
    if "trained" not in r: continue
    v = r["trained"]["values"]
    keys = [k for k in v if k.startswith("layers.8.") or k.startswith("layers.12.") or "blend" in k][:12]
    print(f"- **{a}**: " + "; ".join(f"`{k}`={[round(x,3) for x in v[k][:4]] if isinstance(v[k], list) else round(v[k],3)}" for k in keys))
print("\n### Generations (trained, greedy)\n")
for a, r in d.items():
    if "trained" not in r: continue
    print(f"- **{a}**: " + " / ".join(repr(g[:140]) for g in r["trained"]["generations"]))
for a, r in d.items():
    if "probe_layer14" in r: print(f"\n### Layer-14 probe ({a})\n\n```\n{json.dumps(r['probe_layer14'], indent=1)}\n```")
