#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=11:00:00
#SBATCH --job-name=bb_r56
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/bb_resnet56_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/bb_resnet56_%j.out
set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
BB=/anvil/scratch/x-ashyam2/BayesianBits
export PYTHONPATH=$BB LC_ALL=C.UTF-8 LANG=C.UTF-8
export HF_HUB_OFFLINE=1
PY="$ENV/bin/python"
EPOCHS=${EPOCHS:-2}
MU=${MU:-0.01}
cd "$BB"
"$PY" benchmarks/image_net/gated_mixed_precision.py \
  --experiment=cifar10 --model resnet56_quantized \
  --data-dir /anvil/scratch/x-ashyam2/SQS/Data --method bayesian_bits \
  --gamma-4-init 6 --gamma-8-init 6 --gamma-16-init 6 --gamma-32-init 6 \
  --gating-method l0 \
  --learning-rate 1e-3 --learning-rate-q 1e-3 --learning-rate-s 1e-3 \
  --gating-lambda ${MU} --batch-size 128 --eval-every 1 --epochs ${EPOCHS} \
  --learned-scale=range --include-pruning \
  --optimizer-q Adam --optimizer Adam --optimizer-s Adam \
  --reg-type bop \
  --save-checkpoints --logdir /anvil/scratch/x-ashyam2/SQS/bb_ckpt
echo "BB_EXIT=$?"
