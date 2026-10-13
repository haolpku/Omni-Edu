#!/usr/bin/env bash
#
# Single entry point for the OmniEdu supervised fine-tuning runs.
# Run from the repository root:
#
#   bash training/run_training.sh 9b
#   bash training/run_training.sh 27b --hosts node0,node1 \
#     --env-file training/automation/h20_bond1.env
#
# 4B and 9B train on one node, 27B on two nodes; remaining flags are
# forwarded to training/automation/train.py.
set -euo pipefail

MODEL_KEY="${1:-}"
if [[ -n "$MODEL_KEY" ]]; then
  shift
fi

case "$MODEL_KEY" in
  4b | 9b | 27b) ;;
  *)
    echo "usage: $0 {4b|9b|27b} [train.py options...]" >&2
    exit 2
    ;;
esac

exec bash "training/automation/launch.sh" --model-size "$MODEL_KEY" "$@"
