#!/bin/bash
# Build conda env for SQS, pinned to the author's proven working versions
# (from log/need_save/files/requirements.txt + wandb-metadata.json: py3.12.2, torch2.3.0+cu121).
set -e
set -x

SCRATCH=/anvil/scratch/x-ashyam2
export CONDA_PKGS_DIRS=$SCRATCH/.conda/pkgs
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
ENV=$SCRATCH/envs/sqs
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# 1) Skeleton env (python + pip only)
conda create -y -p "$ENV" python=3.12 pip

PIP="$ENV/bin/pip"
"$ENV/bin/python" -m pip install --upgrade pip

# 2) Torch stack (cu121) — exact match to author's working run
"$PIP" install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 3) Core deps pinned to the saved working environment
"$PIP" install \
  transformers==4.48.1 datasets==3.2.0 accelerate==1.3.0 tokenizers==0.21.0 \
  "mosaicml==0.28.0" torchmetrics==1.6.0 wandb==0.19.4 \
  scikit-learn==1.6.1 kmeans-pytorch==0.3 "numpy==2.0.1" \
  hf_transfer==0.1.9 evaluate sentencepiece "protobuf==4.25.3"

# 4) bitsandbytes (imported by glue_training.py)
"$PIP" install bitsandbytes==0.42.0

# 5) ResNet-side deps (timm + detectors register the cifar-pretrained models)
"$PIP" install timm detectors

# 6) Editable install of the SQS library
"$PIP" install -e "$SCRATCH/SQS/src/SQS"

echo "=== ENV BUILD DONE ==="
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
