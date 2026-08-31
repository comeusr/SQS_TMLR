#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --job-name=r56c10
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/resnet56_c10_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/resnet56_c10_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export SQS_DATA_DIR=$SC/SQS/Data HF_HOME=$SC/.cache/huggingface TORCH_HOME=$SC/.cache/torch
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ENV/bin/python"
hostname; nvidia-smi | head -12

# Paper target (Table): ResNet-56 CIFAR-10, 2-bit (K=4), 50% NZ, 32x, -0.84% from 94.37%.
cd "$SC/SQS/src/resnet"
"$PY" resnet_main.py \
  --network resnet56 --dataset cifar10 --num-classes 10 --mask \
  --batch-size 128 --workers 12 \
  --K 4 --tau 0.001 --lr 5e-5 --weight_decay 5e-7 \
  --duration 30ep --alpha_f 0.007 --seed 428 \
  --init_method k-means --run_name repro_r56_c10_sqs \
  --sample --average_num 10 --prune --prior spike_slab --method SQS \
  --eval_interval 2ep --prune_scale 0.01 --warm_up 5ep \
  --init_sparsity 0.0 --final_sparsity 0.50 --prune_end 20ep --prune_init_lr 0.012 \
  --freeze_weight \
  --save_folder "$SC/SQS/runs/repro_r56_c10"
echo "RESNET56_EXIT=$?"