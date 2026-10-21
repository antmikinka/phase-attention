"""results/scratch/*.json -> markdown tables (stdout)."""
import glob, json, math, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_model.stats import paired_tests
R = [json.load(open(f)) for f in sorted(glob.glob("results/scratch/*.json"))]
ARMS = ["baseline", "phase_kv_fb", "CTRL_shuffled_fb", "phase_head", "phase_track", "chunkrope", "phase_kv"]
by = lambda task, arm: [r for r in R if r["task"] == task and r["arm"] == arm]
ms = lambda v: f"{st.mean(v):.4f}" + (f" ± {st.stdev(v):.4f}" if len(v) > 1 else "")
f2 = lambda p: f"{p:.2g}"

print("## (a) TinyStories byte-level LM (val = 64×256 windows of the valid split)\n")
print("| arm | params | seeds | val bpb (mean ± sd) | byte ppl | eval tok/s | vs baseline: Δbpb, windows better /64, sign p / Wilcoxon p (seed-mean windows) | per-seed Δbpb |")
print("|---|---|---|---|---|---|---|---|")
base = {r["seed"]: r for r in by("lm", "baseline")}
def win_mean(rs):
    return [st.mean(ws) for ws in zip(*[r["window_loss"] for r in rs])]
for arm in ARMS:
    rs = by("lm", arm)
    if not rs: continue
    bpb = [r["val_bpb"] for r in rs]
    cell = per = "-"
    if arm != "baseline" and base:
        common = [r for r in rs if r["seed"] in base]
        if common:
            t = paired_tests(win_mean(common), win_mean([base[r["seed"]] for r in common]))
            cell = f"{t['mean_diff']/math.log(2):+.4f}, {t['a_lower_in']}/64, {f2(t['sign_p'])} / {f2(t['wilcoxon_p'])}"
            per = ", ".join(f"s{r['seed']}: {r['val_bpb']-base[r['seed']]['val_bpb']:+.4f}" for r in common)
    print(f"| {arm} | {rs[0]['params']:,} | {len(rs)} | {ms(bpb)} | {st.mean(math.exp(r['val_loss']) for r in rs):.3f} | "
          f"{st.mean(r['val_tok_per_s'] for r in rs):,.0f} | {cell} | {per} |")

print("\n## (b) Associative recall (10 pairs / 128 tokens, P=4 blocks; chance ≈ 1/32 = 3.1%)\n")
lags = None
for arm in ARMS:
    rs = by("recall", arm)
    if rs: lags = list(rs[0]["recall_acc"]); break
if lags:
    print("Lags in tokens. **Odd number of blocks (4, 12, 28, 60) = pair stored in the phase OPPOSITE the query** (suppressed for kv phase); 8, 16, 32, 64 = same phase.\n")
    print("| arm | params | seeds | " + " | ".join(f"acc@{k}" for k in lags) + " | cross-phase | same-phase | tok/s |")
    print("|---|---|---|" + "---|" * (len(lags) + 3))
    for arm in ARMS:
        rs = by("recall", arm)
        if not rs: continue
        print(f"| {arm} | {rs[0]['params']:,} | {len(rs)} | " + " | ".join(f"{100*st.mean(r['recall_acc'][k] for r in rs):.1f}" for k in lags)
              + f" | {100*st.mean(r['recall_acc_cross_phase'] for r in rs):.1f} | {100*st.mean(r['recall_acc_same_phase'] for r in rs):.1f} | {st.mean(r['val_tok_per_s'] for r in rs):,.0f} |")
    print("\nPer-seed cross-phase / same-phase accuracy:\n")
    for arm in ARMS:
        rs = by("recall", arm)
        if rs: print(f"- {arm}: " + "; ".join(f"s{r['seed']} {100*r['recall_acc_cross_phase']:.1f} / {100*r['recall_acc_same_phase']:.1f}" for r in rs))
    rb = {r["seed"]: r for r in by("recall", "baseline")}
    print("\nPaired tests vs baseline over the 32 held-out eval batches (seed-mean accuracy; long = lags 28–64):\n")
    print("| arm | metric | Δ acc (pp) | batches better/worse | sign p / Wilcoxon p |")
    print("|---|---|---|---|---|")
    def batch_acc(rs, keys):
        out = []
        for i in range(32):
            vals = [st.mean(v for k in keys if (v := r["recall_acc_per_batch"][i][k]) is not None) for r in rs]
            out.append(st.mean(vals))
        return out
    groups = {"cross-phase (4,12,28,60)": [k for k in lags if (int(k) // 4) % 2 == 1],
              "same-phase (8,16,32,64)": [k for k in lags if (int(k) // 4) % 2 == 0],
              "long cross (28,60)": [k for k in lags if int(k) in (28, 60)]}
    for arm in ARMS[1:]:
        rs = [r for r in by("recall", arm) if r["seed"] in rb]
        if not rs: continue
        for gname, keys in groups.items():
            a_, b_ = batch_acc(rs, keys), batch_acc([rb[r["seed"]] for r in rs], keys)
            t = paired_tests(a_, b_)
            print(f"| {arm} | {gname} | {100*t['mean_diff']:+.1f} | {t['a_higher_in']}/{t['a_lower_in']} | {f2(t['sign_p'])} / {f2(t['wilcoxon_p'])} |")

print("\n## Learned phase / gate parameters (mean over blocks; per seed)\n")
for task in ("lm", "recall"):
    for arm in ARMS[1:]:
        for r in by(task, arm):
            pr = r["phase_report"]
            if not pr: continue
            keys = sorted({k for b in pr for k in b if k not in ("block", "mode")})
            vals = {k: st.mean(b[k] for b in pr if k in b) for k in keys}
            print(f"- **{task}/{arm}/s{r['seed']}**: " + ", ".join(f"{k}={v:.3g}" for k, v in vals.items()))

if os.path.exists("results/scratch_bench.json"):
    B = json.load(open("results/scratch_bench.json"))
    print("\n## (c) Throughput and attention work actually computed (forward only, untrained weights, 8 threads; median of runs)\n")
    print("| variant | QK entries computed (rel. baseline causal kernel) | tok/s T=256 | T=1024 | T=2048 |")
    print("|---|---|---|---|---|")
    names = sorted({k.split("/", 1)[1] for k in B}, key=lambda n: list(B).index(f"T256/{n}"))
    for n in names:
        print(f"| {n} | {B[f'T256/{n}']['score_entries_rel_baseline']} | " + " | ".join(f"{st.median(B[f'T{T}/{n}']['tok_per_s_runs']):,.0f}" for T in (256, 1024, 2048)) + " |")
