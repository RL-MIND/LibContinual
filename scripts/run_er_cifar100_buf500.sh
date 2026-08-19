#!/usr/bin/env bash
set -euo pipefail

CONFIG="er_cifar100_buf500"
GPU="${1:-0}"
SEEDS="${2:-0 1 2 3 4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/output/runs/er_cifar100_buf500_${RUN_TAG}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

if [ ! -f "datasets/cifar-100-python/train" ] || [ ! -f "datasets/cifar-100-python/test" ]; then
  echo "ERROR: CIFAR-100 binary files were not found under datasets/cifar-100-python."
  exit 1
fi

echo "Experiment : PaperER baseline, CIFAR-100, buffer=500, CIL"
echo "GPU        : ${GPU}"
echo "Seeds      : ${SEEDS}"
echo "Log dir    : ${LOG_DIR}"

for SEED in ${SEEDS}; do
  LOG_FILE="${LOG_DIR}/seed_${SEED}.log"
  echo "================ seed=${SEED} ================"
  python run_trainer.py --config "${CONFIG}" --device "${GPU}" --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"
done

SUMMARY_FILE="${LOG_DIR}/summary.txt"
{
  echo "Experiment: PaperER CIFAR-100 buffer=500 CIL"
  echo "Seeds: ${SEEDS}"
  echo
  for SEED in ${SEEDS}; do
    echo "----- seed ${SEED} -----"
    grep -E "\[Paper\] Final Average Accuracy|\[Paper\] Final Forgetting|\[Paper\] Expected Calibration Error" "${LOG_DIR}/seed_${SEED}.log" | tail -n 3 || true
  done
} | tee "${SUMMARY_FILE}"
