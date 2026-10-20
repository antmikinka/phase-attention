#!/usr/bin/env bash
# Sequential queue: train only the new layers (frozen Qwen3-0.6B), resumable. 8 threads.
cd "$(dirname "$0")/.."
P=${PY:-/workspace/krystal-qwen/.venv/bin/python}
for arm in hard_restore_either CTRL_hard_restore_either_shuffled hard_restore_either_gw1.0 hard_restore_either_gw0.2 \
           hard_restore_either_gw0.05 hard_alone soft_track chunk4; do
  extra=""; [ "$arm" = "hard_restore_either" ] && extra="--probe"
  $P scripts/train_new_layers.py --arm $arm --steps 200 --threads 8 $extra >> logs/trained_$arm.log 2>&1
  echo "done $arm $(date)"
done
echo ALL_DONE
