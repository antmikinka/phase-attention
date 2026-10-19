#!/usr/bin/env bash
# Harder rivalry variant: at the query, report where the SUPPRESSED progression is now
# (it was hidden by noise for `delay` steps), so the suppressed content must be tracked.
set -u
cd "$(dirname "$0")/.."
P=${PY:-/workspace/krystal-qwen/.venv/bin/python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for arm in baseline phase_fixed phase_all phase_random; do
  $P scripts/train.py --task rivalry --recall extrap --arm $arm --seq 48 --steps 3000 --eval_every 1000 --threads 2 \
     --out results/smoke_rivalry_extrap.json > logs/rivalry_extrap_$arm.log 2>&1
  echo "done rivalry_extrap $arm $(date)"
done
echo EXTRAP_DONE
