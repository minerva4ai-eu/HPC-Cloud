#!/bin/bash

#SBATCH --job-name=cloud-hpc-heavy-common-v1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/heavy-common-%j.out
#SBATCH --error=logs/heavy-common-%j.err

set -euo pipefail

export CPU_THREADS=8
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_heavy_mn5_v1.slurm"
