#!/bin/bash
# Run one scorpio command on one GPU, from the repo root:
#
#   sbatch [--account=... --partition=...] slurm/run.sbatch <command> [overrides...]
#   sbatch slurm/run.sbatch prepare dataset=hmmt shard=0/4
#SBATCH --job-name=scorpio
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# SGLang compiles kernels at runtime, so the job needs the CUDA 13 toolkit.
# Adapt this line to how your cluster provides it.
module load gcc cuda/13.2
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

source .venv/bin/activate
python -m "scorpio.cli.$1" "${@:2}"
