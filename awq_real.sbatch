#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --job-name=awq_real
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/awq_real_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/awq_real_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export HF_HOME=$SC/.cache/huggingface TORCH_HOME=$SC/.cache/torch
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SQS_QWEN05B_PATH=${SQS_QWEN05B_PATH:-$SC/SQS/runs/ft/qwen_sst2/epoch0}
PY="$ENV/bin/python"
cd "$SC/SQS/src/baselines/AWQ"
echo "Base model: $SQS_QWEN05B_PATH"
# Sweep for the SQS-vs-AWQ curve (effective bits vs SST-2 accuracy).
for cfg in "4 128" "4 64" "3 128" "3 64" "2 128"; do
  set -- $cfg
  echo "==================== AWQ w_bit=$1 group=$2 ===================="
  "$PY" awq_real.py --model_name Qwen/Qwen2.5-0.5B --task sst2 \
    --w_bit $1 --group $2 --n_calib 128 --max_len 128
done
echo "AWQ_REAL_EXIT=$?"
