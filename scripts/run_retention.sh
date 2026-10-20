#!/usr/bin/env bash
# Sequential retention-reward queue (resumable; skips finished arms). Commits results after each arm.
cd "$(dirname "$0")/.."
P=${PY:-/workspace/krystal-qwen/.venv/bin/python}
export GIT_AUTHOR_NAME=Anthony GIT_COMMITTER_NAME=Anthony GIT_AUTHOR_EMAIL=67480807+antmikinka@users.noreply.github.com GIT_COMMITTER_EMAIL=67480807+antmikinka@users.noreply.github.com
for arm in hard_restore_aux0.1 CTRL_identity_aux0.1 CTRL_shuffled_aux0.1 STOCK_readout hard_restore_aux0.3 CTRL_identity_aux0.3 \
           hard_restore_aux0.03 CTRL_identity_aux0.03 hard_alone_aux0.1 CTRL_identity_small_aux0.1 hard_restore_detach CTRL_identity_detach; do
  if grep -q "DONE $arm\$" logs/ret_$arm.log 2>/dev/null; then echo "skip $arm"; continue; fi
  $P scripts/train_retention.py --arm $arm >> logs/ret_$arm.log 2>&1
  echo "done $arm $(date)"
  git add results/retention.json results/retention_params logs/ret_* >/dev/null 2>&1
  git commit -q -m "retention: $arm" >/dev/null 2>&1 && git push -q >/dev/null 2>&1
done
echo ALL_DONE
