#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-ider_cifar100_buf500}"
GPU="${2:-0}"
SEED="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Config      : ${CONFIG}"
echo "GPU         : ${GPU}"
echo "Seed        : ${SEED}"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.zeros(1, device="cuda")
    print("cuda sanity:", x)
PY

python run_trainer.py \
  --config "${CONFIG}" \
  --device "${GPU}" \
  --seed "${SEED}"
