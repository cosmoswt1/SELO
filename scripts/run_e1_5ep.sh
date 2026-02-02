#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
EXP_ID="E1"
RUN_DIR="${ROOT_DIR}/runs/${EXP_ID}"
QUAL_DIR="${ROOT_DIR}/outputs/qual/${EXP_ID}"

mkdir -p "${RUN_DIR}" "${QUAL_DIR}"

conda run -n selo python "${ROOT_DIR}/scripts/guard_gpu.py"

conda run -n selo python "${ROOT_DIR}/train_selo_v0.py" \
  --acdc_root "${ACDC_ROOT}" \
  --output_dir "${RUN_DIR}" \
  --seed 1 \
  --epochs 5 \
  --auto_batch \
  --auto_batch_candidates "8,6,4,2,1" \
  --workers 4 \
  --affinity_k 5 \
  --affinity_tau 0.1 \
  --lambda_aff 1.0 \
  --amp

conda run -n selo python "${ROOT_DIR}/scripts/eval_selo_v0.py" \
  --acdc_root "${ACDC_ROOT}" \
  --split val \
  --output_dir "${RUN_DIR}" \
  --qual_dir "${QUAL_DIR}" \
  --ckpt "${RUN_DIR}/adapter.pth" \
  --exp_id "${EXP_ID}" \
  --seed 1 \
  --epochs 5
