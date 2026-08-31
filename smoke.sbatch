#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --job-name=sqs_smoke
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/smoke_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/smoke_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export SQS_DATA_DIR=$SC/SQS/Data
export HF_HOME=$SC/.cache/huggingface
export TORCH_HOME=$SC/.cache/torch
export WANDB_MODE=disabled
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Compute nodes may lack internet; force use of the login-node-populated cache.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="$ENV/bin/python"

echo "=================== NODE INFO ==================="
hostname
nvidia-smi
"$PY" -c "import torch;print('CUDA?',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo ""
echo "=================== RESNET SMOKE (cifar100 / resnet18 / SQS) ==================="
# Tiny subset for a fast end-to-end pass; exercises DGMS conv transform,
# k-means GMM init, SQS pruning algorithm, composer train + eval loop.
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
unset SQS_MAX_TRAIN SQS_MAX_EVAL

echo ""
echo "=================== LLM SMOKE (Qwen2.5-0.5B / sst2 / SQS) ==================="
cd "$SC/SQS/src/glue"
"$PY" glue_training.py \
  --task_name sst2 --dataset_name glue \
  --model_name Qwen/Qwen2.5-0.5B --method SQS \
  --K 16 --tau 0.001 --weight_decay 5e-7 --prune \
  --lr 5e-6 --prune_lr 0.01 --duration 1 --alpha_f 0.01 --seed 428 \
  --init_method k-means --optimizer sgd --sigma 3 \
  --prune_scale 0.01 --prune_start 0.0 --prune_end 0.9 \
  --init_sparsity 0.0 --final_sparsity 0.75 \
  --batch_size 8 --eval_batch_size 32 --max_seq_length 128 \
  --sample_data_num 128 --precision fp32 \
  --run_name smoke_qwen --base_dir "$SC/SQS/runs/llm_smoke" \
  --save_folder "$SC/SQS/runs/llm_smoke/ckpt"
echo "LLM_EXIT=$?"

echo ""
echo "=================== SMOKE DONE ==================="
