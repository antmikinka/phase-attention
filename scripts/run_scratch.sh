#!/usr/bin/env bash
# usage: run_scratch.sh <task> <threads> <seeds...>  — sequential, resumable, commits after each run
cd "$(dirname "$0")/.."
P=${PY:-/home/box/venvs/phase/bin/python}
task=$1; th=$2; shift 2
export GIT_AUTHOR_NAME=Anthony GIT_COMMITTER_NAME=Anthony GIT_AUTHOR_EMAIL=67480807+antmikinka@users.noreply.github.com GIT_COMMITTER_EMAIL=67480807+antmikinka@users.noreply.github.com
for seed in "$@"; do
  for arm in baseline phase_kv_fb CTRL_shuffled_fb phase_head phase_track chunkrope phase_kv; do
    $P scripts/scratch_train.py --task $task --arm $arm --seed $seed --threads $th >> logs/scratch/${task}_${arm}_s${seed}.log 2>&1
    echo "done $task $arm s$seed $(date)"
    ( flock 9; git add results/scratch logs/scratch >/dev/null 2>&1; git commit -q -m "scratch: $task $arm s$seed" >/dev/null 2>&1 && git push -q >/dev/null 2>&1 ) 9>/tmp/phase_git.lock
  done
done
echo ALL_DONE
