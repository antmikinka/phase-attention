#!/usr/bin/env bash
cd "$(dirname "$0")/.."
P=${PY:-/workspace/krystal-qwen/.venv/bin/python}
{ cat docs/RESULTS_SMOKE_HEAD.md; $P scripts/summarize.py; } > RESULTS_SMOKE.md
