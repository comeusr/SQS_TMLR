#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --job-name=sqs_llm
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/llm_smoke_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/llm_smoke_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
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
echo "=================== LLM SMOKE DONE ==================="
