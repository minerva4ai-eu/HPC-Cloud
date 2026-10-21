#!/bin/bash

#SBATCH --job-name=cloud-hpc-heavy-v1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=logs/heavy-%j.out
#SBATCH --error=logs/heavy-%j.err

set -euo pipefail

REMOTE_ROOT="${PROJECT_PATH:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"

SCRIPT="${REMOTE_ROOT}/scripts/train_heavy_ai_v1.py"

SIF="${CONTAINER_IMAGE_PATH:-${REMOTE_ROOT}/containers/tensorflow-2.21.0-cpu.sif}"

RESULT_DIR="${REMOTE_ROOT}/results/job-${SLURM_JOB_ID}"

RESULT_FILE="${RESULT_DIR}/result_heavy.json"

mkdir -p "${RESULT_DIR}"

echo "============================================="
echo "Cloud-HPC Synthetic Heavy MN5 Validation"
echo "============================================="

date -u +"UTC_JOB_START=%Y-%m-%dT%H:%M:%S.%NZ"

echo
echo "========== SLURM =========="

echo "JOB_ID=${SLURM_JOB_ID}"
echo "JOB_NAME=${SLURM_JOB_NAME}"
echo "NODELIST=${SLURM_JOB_NODELIST}"
echo "CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

echo
echo "========== HOST =========="

hostname
uname -m

echo
echo "========== CPU =========="

lscpu | grep -E \
'CPU\(s\):|Model name:|Thread\(s\) per core:|Core\(s\) per socket:|Socket\(s\):'

echo
echo "========== MEMORY =========="

free -h

echo
echo "========== ARTIFACT INTEGRITY =========="

SCRIPT_SHA256="$(sha256sum "${SCRIPT}" | awk '{print $1}')"

SIF_SHA256="$(sha256sum "${SIF}" | awk '{print $1}')"

echo "SCRIPT_SHA256=${SCRIPT_SHA256}"
echo "SIF_SHA256=${SIF_SHA256}"

EXPECTED_SCRIPT_SHA256="a1955e873fe24cb39633f914562d96435bde39c92a5cfc7726c894a1ce520484"

EXPECTED_SIF_SHA256="4baf239bc26d7563bb02f6bc105989f09033ac5a6d5b79f5d68a7fca7016d45f"

if [ "${SCRIPT_SHA256}" != "${EXPECTED_SCRIPT_SHA256}" ]; then
    echo "SCRIPT_CHECKSUM=FAIL"
    exit 10
fi

echo "SCRIPT_CHECKSUM=PASS"

if [ "${SIF_SHA256}" != "${EXPECTED_SIF_SHA256}" ]; then
    echo "SIF_CHECKSUM=FAIL"
    exit 11
fi

echo "SIF_CHECKSUM=PASS"

echo
echo "========== FRAMEWORK =========="

SINGULARITY_BIN="${SINGULARITY_BIN:-$(command -v singularity || true)}"

if [ ! -x "${SINGULARITY_BIN}" ]; then
    echo "SINGULARITY_BINARY=NOT_FOUND"
    exit 12
fi

echo "SINGULARITY_BIN=${SINGULARITY_BIN}"

"${SINGULARITY_BIN}" --version

echo "SINGULARITY_RUNTIME=PASS"

echo
echo "========== HEAVY AI EXECUTION =========="

export PYTHONHASHSEED=20260813

export CPU_THREADS="${CPU_THREADS:-16}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"

export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-${CPU_THREADS}}"

export TF_NUM_INTEROP_THREADS=2

export RESULT_PATH="${RESULT_FILE}"

AI_START="$(date +%s)"

set +e

"${SINGULARITY_BIN}" exec \
  "${SIF}" \
  python3 "${SCRIPT}"

AI_EXIT_CODE=$?

set -e

AI_END="$(date +%s)"

AI_SECONDS=$((AI_END - AI_START))

echo
echo "AI_EXIT_CODE=${AI_EXIT_CODE}"
echo "AI_WALL_SECONDS=${AI_SECONDS}"

if [ "${AI_EXIT_CODE}" -ne 0 ]; then
    echo "HEAVY_MN5_EXECUTION=FAIL"
    exit "${AI_EXIT_CODE}"
fi

echo
echo "========== RESULT =========="

cat "${RESULT_FILE}"

echo
echo "========== RESULT INTEGRITY =========="

RESULT_SHA256="$(sha256sum "${RESULT_FILE}" | awk '{print $1}')"

echo "RESULT_SHA256=${RESULT_SHA256}"

python3 - "${RESULT_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as fh:
    result = json.load(fh)

assert result["experiment"] == "cloud-hpc-synthetic-heavy-v1"
expected_cpu_threads = int(os.environ["CPU_THREADS"])
assert result["cpu_threads"] == expected_cpu_threads
assert result["test_accuracy"] >= 0.95

print(
    "VALIDATED_TEST_ACCURACY="
    + str(result["test_accuracy"])
)

print(
    "VALIDATED_CPU_THREADS="
    + str(result["cpu_threads"])
)

print("RESULT_ACCEPTANCE=PASS")
PY

date -u +"UTC_JOB_END=%Y-%m-%dT%H:%M:%S.%NZ"

echo "HEAVY_MN5_EXECUTION=PASS"
