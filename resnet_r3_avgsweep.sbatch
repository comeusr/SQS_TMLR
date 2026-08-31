#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --job-name=r18_r3
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/resnet_r3_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/resnet_r3_%j.out

# R3: after training the SQS ResNet-18/CIFAR-100, sweep the Bayesian-average sample
# count M in {1,2,5,10,20,50} at eval (--avg_sweep) -> accuracy vs M. Same recipe as the
# repro so the trained model is representative.
set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export SQS_DATA_DIR=$SC/SQS/Data HF_HOME=$SC/.cache/huggingface TORCH_HOME=$SC/.cache/torch
export WANDB_MODE=offline WANDB_DIR=$SC/SQS/wandb WANDB_PROJECT=SQS-repro
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ENV/bin/python"
cd "$SC/SQS/src/resnet"
"$PY" resnet_main.py \
  --network resnet18 --dataset cifar100 --num-classes 100 --mask \
  --batch-size 128 --workers 12 \
  --K 16 --tau 0.001 --lr 5e-5 --weight_decay 5e-7 \
  --duration 11ep --alpha_f 0.007 --seed 428 \
  --init_method k-means --run_name r18_c100_r3 \
  --sample --average --average_num 10 --avg_sweep \
  --prune --prior spike_slab --method SQS \
  --eval_interval 1ep --prune_scale 0.01 --warm_up 5ep \
  --init_sparsity 0.0 --final_sparsity 0.50 --prune_end 9ep --prune_init_lr 0.012 \
  --freeze_weight \
  --save_folder "$SC/SQS/runs/r18_r3"
echo "RESNET_R3_EXIT=$?"
