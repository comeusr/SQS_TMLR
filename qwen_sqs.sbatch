#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --job-name=sqs_qwen
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/qwen_sqs_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/qwen_sqs_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export HF_HOME=$SC/.cache/huggingface TORCH_HOME=$SC/.cache/torch
# wandb: Anvil compute nodes are OFFLINE, so log locally then `wandb sync` from the
# login node (sync_wandb.sh). offline mode writes $WANDB_DIR/wandb/offline-run-*.
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_DIR=$SC/SQS/wandb WANDB_PROJECT=SQS-Qwen
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Compress the SST-2 fine-tuned base (set by launcher to the best epoch dir):
export SQS_QWEN05B_PATH=${SQS_QWEN05B_PATH:-$SC/SQS/runs/ft/qwen_sst2/epoch2}
PY="$ENV/bin/python"
hostname; nvidia-smi | head -12
echo "Base model: $SQS_QWEN05B_PATH"; ls "$SQS_QWEN05B_PATH" 2>/dev/null | head

# Paper target (Table 3b): Qwen2.5-0.5B SST-2, 6-bit (K=64), 50% NZ, 11x, -2.46% from 92.60%.
K=${K:-64}; SPARS=${SPARS:-0.50}; RUN=${RUN:-sqs_qwen_k${K}_s${SPARS}}
cd "$SC/SQS/src/glue"
"$PY" glue_training.py \
  --task_name sst2 --dataset_name glue --model_name Qwen/Qwen2.5-0.5B --method SQS \
  --K ${K} --tau 0.001 --weight_decay 5e-7 --prune \
  --lr 1e-5 --prune_lr 0.005 --duration 2 --alpha_f 0.01 --seed 428 \
  --init_method k-means --optimizer sgd --sigma 3 \
  --prune_scale 0.01 --prune_start 0.0 --prune_end 0.25 \
  --init_sparsity 0.0 --final_sparsity ${SPARS} \
  --batch_size 16 --eval_batch_size 64 --max_seq_length 128 \
  --sample_data_num 40000 --precision fp32 --eval_steps 200 \
  --run_name ${RUN} --project_name SQS-Qwen --base_dir "$SC/SQS/runs/sqs_qwen" \
  --save_folder "$SC/SQS/runs/sqs_qwen/${RUN}" ${WANDA} ${NO_OUTLIER} ${PRIOR_FIX}
echo "SQS_QWEN_EXIT=$?"