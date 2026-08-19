#!/usr/bin/env bash
set -euo pipefail

CONFIG="ider_cifar100_buf500_til"
GPU="${1:-0}"
SEED="${2:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/train_one_config.sh" "${CONFIG}" "${GPU}" "${SEED}"
