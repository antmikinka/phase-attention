#!/usr/bin/env python
"""(c) Throughput and attention work actually skipped. Forward-only tok/s at several T for: baseline,
hard phase kv via mask (computes everything, masks half), hard phase kv via split (really computes only
same-phase pairs), soft phase kv + feedback (computes main + suppressed partitions). -> results/scratch_bench.json"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import scratch_train as st
from phase_model import ModelConfig, PhaseLM, layer_specs

torch.set_num_threads(int(os.environ.get("THREADS", 8)))
VARIANTS = {"baseline": [], "hard_kv_mask": [{"mixer": {"type": "phase", "hard": True}}],
            "hard_kv_split(skip)": [{"mixer": {"type": "phase", "hard": True, "skip": True}}],
            "soft_kv": st.ARMS["phase_kv"], "soft_kv_feedback": st.ARMS["phase_kv_fb"], "phase_head": st.ARMS["phase_head"]}
res = {}
for rep in range(int(os.environ.get("REPS", 1))):
  for T in (256, 1024, 2048):
    B = max(1, 4096 // T)
    for name, rules in VARIANTS.items():
        torch.manual_seed(0)
        m = PhaseLM(ModelConfig(d_model=st.D, n_layers=st.L, n_heads=st.H, mlp_hidden=st.BASE_HIDDEN, blocks=layer_specs(st.L, rules))).eval()
        x = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            m(x); t = time.time(); n = 0
            while time.time() - t < 6:
                m(x); n += 1
        tps = n * B * T / (time.time() - t)
        # attention score pairs actually computed per layer per sequence (QK^T entries incl. masked ones)
        # QK^T score entries actually computed per layer, relative to the baseline's causal kernel (~T^2/2):
        # dense masked kernels compute T^2; the split computes two causal halves 2*(T/2)^2/2 = T^2/4.
        rel = {"baseline": 1.0, "hard_kv_mask": 2.0, "hard_kv_split(skip)": 0.5, "soft_kv": 2.0,
               "soft_kv_feedback": 4.0, "phase_head": 1.0}[name]
        res.setdefault(f"T{T}/{name}", {"tok_per_s_runs": [], "score_entries_rel_baseline": rel})["tok_per_s_runs"].append(round(tps))
        print(T, name, round(tps), rel, flush=True)
json.dump(res, open("results/scratch_bench.json", "w"), indent=1)
