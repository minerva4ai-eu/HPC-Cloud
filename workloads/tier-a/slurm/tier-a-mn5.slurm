#!/bin/bash
#SBATCH --job-name=cloud-hpc-tier-a-mn5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/tier-a-mn5-%j.out
#SBATCH --error=logs/tier-a-mn5-%j.err

set -euo pipefail

BASE="${PROJECT_PATH:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"

SCRIPT="$BASE/scripts/train_cpu_ai_v1.py"
SIF="${CONTAINER_IMAGE_PATH:-$BASE/containers/tensorflow-2.21.0-cpu.sif}"

EXPECTED_SCRIPT_SHA="99100ee2e0609551d392399b3c7ba3625250a0ccf23b2482ddbcc2a0d6476f17"
EXPECTED_SIF_SHA="4baf239bc26d7563bb02f6bc105989f09033ac5a6d5b79f5d68a7fca7016d45f"

RUN_DIR="$BASE/results/job-${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"

echo "============================================="
echo "Cloud-HPC Tier-A MN5 Migration Experiment"
echo "============================================="

echo "UTC_JOB_SCRIPT_START=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

echo
echo "========== SLURM ENVIRONMENT =========="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "JOB_NAME=${SLURM_JOB_NAME}"
echo "ACCOUNT=${SLURM_JOB_ACCOUNT:-unknown}"
echo "QOS=${SLURM_JOB_QOS:-unknown}"
echo "PARTITION=${SLURM_JOB_PARTITION:-unknown}"
echo "NODELIST=${SLURM_JOB_NODELIST:-unknown}"
echo "CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unknown}"
echo "CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unknown}"

echo
echo "========== COMPUTE NODE =========="
hostname
hostname -f
uname -m

echo
echo "========== OS =========="
grep -E '^(NAME|VERSION|PRETTY_NAME)=' /etc/os-release

echo
echo "========== CPU =========="
lscpu | grep -E \
'CPU\(s\)|Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA'

echo
echo "========== MEMORY =========="
free -h

echo
echo "========== ARTIFACT INTEGRITY =========="

SCRIPT_SHA=$(sha256sum "$SCRIPT" | awk '{print $1}')
SIF_SHA=$(sha256sum "$SIF" | awk '{print $1}')

echo "script_sha256=$SCRIPT_SHA"
echo "sif_sha256=$SIF_SHA"

if [[ "$SCRIPT_SHA" != "$EXPECTED_SCRIPT_SHA" ]]; then
    echo "SCRIPT_CHECKSUM=FAIL"
    exit 20
fi

if [[ "$SIF_SHA" != "$EXPECTED_SIF_SHA" ]]; then
    echo "SIF_CHECKSUM=FAIL"
    exit 21
fi

echo "SCRIPT_CHECKSUM=PASS"
echo "SIF_CHECKSUM=PASS"

echo
echo "========== SINGULARITY =========="

if [[ -n "${SINGULARITY_MODULE:-}" ]]; then
    module load "$SINGULARITY_MODULE"
fi

command -v singularity
singularity --version

echo
echo "========== CONTAINER FRAMEWORK =========="

singularity exec \
    --cleanenv \
    "$SIF" \
    python3 -c \
'import tensorflow as tf, numpy as np; print("tensorflow_version="+tf.__version__); print("numpy_version="+np.__version__)'

echo
echo "========== AI WORKLOAD START =========="

echo "AI_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

set +e

srun \
    --ntasks=1 \
    --cpus-per-task=4 \
    --cpu-bind=cores \
    singularity exec \
        --cleanenv \
        --bind "$BASE/scripts:/work:ro" \
        --bind "$RUN_DIR:/results" \
        "$SIF" \
        env \
            PYTHONHASHSEED=20260813 \
            OMP_NUM_THREADS=4 \
            TF_NUM_INTRAOP_THREADS=4 \
            TF_NUM_INTEROP_THREADS=1 \
            python3 /work/train_cpu_ai_v1.py

AI_RC=$?

set -e

echo "AI_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
echo "AI_EXIT_CODE=$AI_RC"

if [[ -f "$RUN_DIR/result.json" ]]; then
    echo
    echo "========== RESULT.JSON =========="
    cat "$RUN_DIR/result.json"

    echo
    echo "========== RESULT CHECKSUM =========="
    sha256sum "$RUN_DIR/result.json"
else
    echo "RESULT_JSON=MISSING"
fi

echo
echo "UTC_JOB_SCRIPT_END=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

if [[ "$AI_RC" -ne 0 ]]; then
    echo "MN5_TIER_A=FAIL"
    exit "$AI_RC"
fi

echo "MN5_TIER_A=PASS"
