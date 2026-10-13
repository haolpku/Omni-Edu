#!/usr/bin/env bash
#
# Per-node launcher for the two-node 27B run.  Start one copy on each node from
# the repository root, and start the second node before the master node:
#
#   # node 1
#   NODE_RANK=1 MASTER_ADDR=<bond1 address of node 0> bash training/launch_27b_node.sh
#   # node 0
#   NODE_RANK=0 MASTER_ADDR=<bond1 address of node 0> bash training/launch_27b_node.sh
#
# MASTER_ADDR and MASTER_PORT must be identical on both nodes; NODE_RANK must
# differ.  Output is appended to training/logs/27b_node<NODE_RANK>.log.
set -euo pipefail

NODE_RANK="${NODE_RANK:?NODE_RANK is required (0 on the master node, 1 on the second node)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required (bond1 address of the master node)}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ENV_FILE="${ENV_FILE:-training/automation/h20_bond1.env}"
LOG="${LOG:-training/logs/27b_node${NODE_RANK}.log}"

case "$NODE_RANK" in
  0 | 1) ;;
  *)
    echo "NODE_RANK must be 0 or 1 for the two-node 27B run, got $NODE_RANK" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$LOG")"

export NNODES=2
export NODE_RANK
export NPROC_PER_NODE
export MASTER_ADDR MASTER_PORT
export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

echo "node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT log=$LOG" >>"$LOG"

exec bash training/automation/worker.sh \
  --model-size 27b \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --env-file "$ENV_FILE" \
  "$@" >>"$LOG" 2>&1
