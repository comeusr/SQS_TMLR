#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --job-name=ft_qwen
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/ft_qwen_%j.out
#SBATCH --error=/anvil/scratch/x-ashyam2/SQS/ft_qwen_%j.out

set -x
ENV=/anvil/scratch/x-ashyam2/envs/sqs
SC=/anvil/scratch/x-ashyam2
export HF_HOME=$SC/.cache/huggingface
export TORCH_HOME=$SC/.cache/torch
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ENV/bin/python"

hostname; nvidia-smi | head -12

echo "################## VALIDATE SPARSITY-GATE FIX ##################"
"$PY" "$SC/SQS/validate_gate.py"; echo "GATE_VALIDATE_EXIT=$?"

echo "################## FINE-TUNE Qwen2.5-0.5B on SST-2 (normal) ##################"
cd "$SC/SQS/src/glue"
"$PY" glue_training.py \
  --task_name sst2 --dataset_name glue --model_name Qwen/Qwen2.5-0.5B --normal \
  --optimizer adam --lr 2e-5 --weight_decay 0.0 \
  --duration 3 --batch_size 32 --eval_batch_size 64 --max_seq_length 128 \
  --sample_data_num 100000 --precision fp32 --eval_steps 500 \
  --run_name ft_qwen_sst2 --base_dir "$SC/SQS/runs/ft" \
  --save_folder "$SC/SQS/runs/ft/qwen_sst2"
echo "FT_EXIT=$?"
echo "################## FT DONE ##################"
ls -la "$SC/SQS/runs/ft/qwen_sst2/"