#!/bin/bash -l
#SBATCH -A cis260089-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --job-name=probe
#SBATCH --output=/anvil/scratch/x-ashyam2/SQS/probe_%j.out
hostname; nvidia-smi | head -3; echo PROBE_OK
