#!/usr/bin/env python
"""Summarize results/smoke_*.json into markdown tables (final eval of each arm)."""
import json, os, sys

def best_stream(ev, key):
    vals = [v for k, v in ev.items() if k.startswith(key + "_stream")]
    return max(vals) if vals else None

def fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{100*x:.1f}" if pct else f"{x:.3f}"

out = []
for task in ["rivalry", "rivalry_extrap"]:
    p = f"results/smoke_{task}.json"
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    out.append(f"\n### {task} (final eval, 512 held-out sequences; chance = 6.25%)\n")
    out.append("| arm | params | steps | dominant report acc | suppressed query acc | probe: suppressed next (z / best stream) | probe: suppressed during noise (z / best stream) | probe: dominant next (z) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, r in R.items():
        ev = r["evals"][-1]
        out.append(f"| {r['arm']} | {r['n_params']:,} | {ev['step']} | {fmt(ev['dom_acc'])} | {fmt(ev['recall_acc'])} | "
                   f"{fmt(ev['probe_sup_next_final'])} / {fmt(best_stream(ev,'probe_sup_next'))} | "
                   f"{fmt(ev['probe_sup_delay_final'])} / {fmt(best_stream(ev,'probe_sup_delay'))} | {fmt(ev['probe_dom_next_final'])} |")
p = "results/smoke_lm.json"
if os.path.exists(p):
    R = json.load(open(p))
    out.append("\n### byte-level TinyStories LM (val bpb, 64×128 bytes; reversal = context reversed inside chunks of C bytes, scored at chunk ends)\n")
    out.append("| arm | params | steps | val bpb | C=2 intact→reversed (ratio) | C=4 | C=8 |")
    out.append("|---|---|---|---|---|---|---|")
    for k, r in R.items():
        ev = r["evals"][-1]
        cells = [f"{ev[f'rev{C}_intact_bpb']:.3f}→{ev[f'rev{C}_reversed_bpb']:.3f} ({ev[f'rev{C}_ratio']:.2f}×)" for C in (2, 4, 8)]
        out.append(f"| {r['arm']} | {r['n_params']:,} | {ev['step']} | {ev['val_bpb']:.3f} | " + " | ".join(cells) + " |")
print("\n".join(out))
