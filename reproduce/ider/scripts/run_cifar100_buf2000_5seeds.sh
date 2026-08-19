#!/usr/bin/env bash
set -euo pipefail

CONFIG="ider_cifar100_buf2000"
GPU="${1:-0}"
SEEDS="${2:-0 1 2 3 4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/output/runs/cifar100_buf2000_5seeds_${RUN_TAG}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

echo "Experiment : CIFAR100, buffer=2000, CIL, 5 seeds"
echo "Config     : ${CONFIG}"
echo "GPU        : ${GPU}"
echo "Seeds      : ${SEEDS}"
echo "Log dir    : ${LOG_DIR}"

if [ ! -f "datasets/cifar-100-python/train" ] || [ ! -f "datasets/cifar-100-python/test" ]; then
  echo "ERROR: CIFAR100 binary files were not found under datasets/cifar-100-python."
  exit 1
fi

for SEED in ${SEEDS}; do
  LOG_FILE="${LOG_DIR}/seed_${SEED}.log"
  echo "================ seed=${SEED} ================"
  python run_trainer.py --config "${CONFIG}" --device "${GPU}" --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"
done

SUMMARY_FILE="${LOG_DIR}/summary.txt"
{
  echo "Experiment: CIFAR100 buffer=2000 CIL"
  echo "Config: ${CONFIG}"
  echo "GPU: ${GPU}"
  echo "Seeds: ${SEEDS}"
  echo
  echo "Final paper metrics by seed:"
  for SEED in ${SEEDS}; do
    LOG_FILE="${LOG_DIR}/seed_${SEED}.log"
    echo "----- seed ${SEED} -----"
    grep -E "\[Paper\] Final Average Accuracy|\[Paper\] Final Forgetting|\[Paper\] Expected Calibration Error|Last Average Acc|Forgetting:|Backward Transfer:" "${LOG_FILE}" | tail -n 8 || true
  done
} | tee "${SUMMARY_FILE}"
