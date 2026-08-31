#!/bin/bash -l
# Sync offline wandb runs to the cloud dashboard. Run this ON THE LOGIN NODE
# (compute nodes have no internet). Safe to re-run; already-synced runs are skipped.
ENV=/anvil/scratch/x-ashyam2/envs/sqs
export WANDB_DIR=/anvil/scratch/x-ashyam2/SQS/wandb
cd "$WANDB_DIR/wandb" || exit 1
for d in offline-run-*; do
  [ -d "$d" ] || continue
  echo "=== syncing $d ==="
  "$ENV/bin/wandb" sync "$d"
done
echo "done. view at https://wandb.ai/  (project SQS-Qwen / your resnet project)"
