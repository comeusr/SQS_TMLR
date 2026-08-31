#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --job-name=sqs_resnet
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/resnet_smoke_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/resnet_smoke_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export SQS_DATA_DIR=$SC/SQS/Data
export HF_HOME=$SC/.cache/huggingface
export TORCH_HOME=$SC/.cache/torch
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="$ENV/bin/python"

hostname; nvidia-smi
"$PY" -c "import torch;print('CUDA?',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "=================== RESNET SMOKE (cifar100 / resnet18 / SQS) ==================="
export SQS_MAX_TRAIN=512
export SQS_MAX_EVAL=512
cd "$SC/SQS/src/resnet"
"$PY" resnet_main.py \
  --network resnet18 --dataset cifar100 --num-classes 100 --mask \
  --batch-size 128 --workers 8 \
  --K 16 --tau 0.001 --lr 5e-5 --weight_decay 5e-7 \
  --duration 1ep --alpha_f 0.007 --seed 428 \
  --init_method k-means --run_name smoke_resnet \
  --sample --average_num 2 --prune --prior spike_slab --method SQS \
  --eval_interval 1ep --prune_scale 0.01 --warm_up 0ep \
  --init_sparsity 0.0 --final_sparsity 0.70 --prune_end 1ep --prune_init_lr 0.012 \
  --freeze_weight \
  --save_folder "$SC/SQS/runs/smoke_resnet"
echo "RESNET_EXIT=$?"
echo "=================== RESNET SMOKE DONE ==================="
