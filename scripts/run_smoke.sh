#!/usr/bin/env bash
# Smoke runs: 1 seed, small steps, 2 torch threads (CPU is shared with another job).
set -u
cd "$(dirname "$0")/.."
P=${PY:-/workspace/krystal-qwen/.venv/bin/python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for arm in baseline phase_fixed phase_all phase_random phase_fixed_noinhib baseline_2x; do
  $P scripts/train.py --task rivalry --arm $arm --seq 48 --steps 3000 --eval_every 1000 --threads 2 \
     --out results/smoke_rivalry.json > logs/rivalry_$arm.log 2>&1
  echo "done rivalry $arm $(date)"
done
for arm in baseline phase_fixed phase_all coarse4; do
  $P scripts/train.py --task lm --arm $arm --seq 128 --steps 1200 --eval_every 400 --threads 2 \
     --out results/smoke_lm.json > logs/lm_$arm.log 2>&1
  echo "done lm $arm $(date)"
done
echo ALL_DONE
