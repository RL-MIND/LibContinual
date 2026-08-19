#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train_one_config.sh <config_name> [gpu] [seed]"
  echo "Example: bash scripts/train_one_config.sh ider_cifar100_buf500 0 4"
  exit 1
fi

CONFIG="$1"
GPU="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/output/runs/single_${CONFIG}_seed${SEED}_${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

echo "Project root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG}"
echo "GPU          : ${GPU}"
echo "Seed         : ${SEED}"
echo "Log file     : ${LOG_FILE}"

python run_trainer.py \
  --config "${CONFIG}" \
  --device "${GPU}" \
  --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"

echo
echo "Final metrics:"
grep -E "\[Paper\] Final Average Accuracy|\[Paper\] Final Forgetting|\[Paper\] Expected Calibration Error|Last Average Acc|Forgetting:|Backward Transfer:" "${LOG_FILE}" | tail -n 8 || true
