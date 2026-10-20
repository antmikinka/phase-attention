#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while ! grep -q ALL_DONE logs/trained_queue.log; do sleep 30; done
/workspace/krystal-qwen/.venv/bin/python scripts/train_new_layers.py --arm CTRL_identity_trainable --steps 200 --threads 8 >> logs/trained_CTRL_identity_trainable.log 2>&1
echo CTRL_DONE
