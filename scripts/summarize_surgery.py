#!/usr/bin/env python
import json, sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/surgery.json"))
stock = next((r for r in R.values() if r["variant"] == "stock"), None)
print("| variant | layers / settings | WikiText-2 ppl | Δ vs stock | prefill tok/s | decode tok/s | greedy reply to 'why is the sky blue' (first 90 chars) |")
print("|---|---|---|---|---|---|---|")
for r in R.values():
    rules = "; ".join(f"{x['layers']}: " + ", ".join(f"{k}={v}" for k, v in (x.get('attention') or x.get('position') or x.get('extra')).items()) for x in r["rules"]) or "–"
    d = f"{100*(r['ppl']/stock['ppl']-1):+.1f}%" if stock else ""
    g = r["generations"][0][:90].replace("\n", " ").replace("|", "/")
    print(f"| {r['variant']} | {rules} | {r['ppl']:.2f} | {d} | {r['prefill_tps']:.0f} | {r['decode_tps']:.1f} | {g} |")
print("\n**Chunk-reversal robustness** (ppl on chunk-end targets, first 8 windows; context reversed inside chunks of C)\n")
print("| variant | C | intact | reversed | ratio |\n|---|---|---|---|---|")
for r in R.values():
    if "reversal" in r:
        v = r["reversal"]; print(f"| {r['variant']} | {v['chunk']} | {v['intact']:.1f} | {v['reversed']:.1f} | {v['ratio']:.1f}× |")
print("\n**Suppressed-stream decoding** (ridge probe at layer 14, 8 windows, top-40 token classes, 60/40 split)\n")
print("| variant | target | majority | from suppressed buffer | from access stream |\n|---|---|---|---|---|")
for r in R.values():
    if "probe" in r:
        for t in ("token_at_last_suppressed_pos", "token_at_last_access_pos"):
            v = r["probe"][t]
            print(f"| {r['variant']} | {t} | {100*v['majority_baseline']:.1f}% | {100*v['from_suppressed_buffer']:.1f}% | {100*v['from_access_stream']:.1f}% |")
for r in R.values():
    if "vs_hf" in r:
        print(f"\n{r['variant']} vs HF Qwen3ForCausalLM (64 tokens): {r['vs_hf']}")
