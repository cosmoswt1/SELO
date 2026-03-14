#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/exp/CKA/sfda_stage3_cka_g67_w16_l24/eval_stage3_cka.py"

ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
RUN_ID="${RUN_ID:=v1-topk-updateloss-binarygate}"
SPLIT="${SPLIT:-val}"
RUN_DIR="${RUN_DIR:=${ROOT_DIR}/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs/${RUN_ID}}"
CKPT="${CKPT:-${RUN_DIR}/adapter.pth}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/eval_${SPLIT}}"
TEST_GT_DIR="${TEST_GT_DIR:-}"

CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  fi
fi

if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
  echo "[run_eval] conda binary not found." >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[GPU 체크 실패] nvidia-smi 실행 실패" >&2
  echo "확인 커맨드:" >&2
  echo "  nvidia-smi" >&2
  echo "  python -c \"import torch; print(torch.cuda.is_available())\"" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  "${CONDA_BIN}" run -n selo python -u "${SCRIPT_PATH}"
  --acdc_root "${ACDC_ROOT}"
  --split "${SPLIT}"
  --resize 1080
  --ckpt "${CKPT}"
  --output_dir "${OUT_DIR}"
)

if [ -n "${TEST_GT_DIR}" ]; then
  CMD+=(--test_gt_dir "${TEST_GT_DIR}")
fi

if [ "${#}" -gt 0 ]; then
  CMD+=("$@")
fi

echo "[run_eval] OUT_DIR=${OUT_DIR}"
echo "[run_eval] command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
