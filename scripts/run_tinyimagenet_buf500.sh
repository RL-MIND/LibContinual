#!/usr/bin/env bash
set -euo pipefail

CONFIG="ider_tinyimagenet_buf500"
GPU="${1:-0}"
SEEDS="${2:-0 1 2 3 4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/output/runs/tinyimagenet_buf500_${RUN_TAG}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

if [ ! -f "datasets/TINYIMG/processed/x_train_01.npy" ] || [ ! -f "datasets/TINYIMG/processed/y_val_20.npy" ]; then
  echo "ERROR: processed TinyImageNet files were not found under datasets/TINYIMG/processed."
  echo "Run: python scripts/download_tinyimagenet_processed.py"
  exit 1
fi

echo "Experiment : IDER, TinyImageNet, buffer=500, CIL"
echo "Config     : ${CONFIG}"
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
  echo "Experiment: IDER TinyImageNet buffer=500 CIL"
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
