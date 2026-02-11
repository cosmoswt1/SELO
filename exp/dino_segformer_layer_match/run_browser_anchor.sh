#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/exp/dino_segformer_layer_match/browser_anchor_server.py"

ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/exp/dino_segformer_layer_match/browser_picker}"
CONDITION="${CONDITION:-night}"
INCLUDE_REF="${INCLUDE_REF:-0}"
IMAGE_INDEX="${IMAGE_INDEX:-0}"
RESIZE_SHORT="${RESIZE_SHORT:-1072}"
SQUARE_CROP_SIZE="${SQUARE_CROP_SIZE:-1072}"
VIZ_GRID_SIZE="${VIZ_GRID_SIZE:-67}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
AMP="${AMP:-1}"
GPU_RETRY="${GPU_RETRY:-8}"
GPU_RETRY_SLEEP_SEC="${GPU_RETRY_SLEEP_SEC:-2}"
DRY_RUN="${DRY_RUN:-0}"

CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  fi
fi

if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
  echo "[run_browser_anchor] conda binary not found." >&2
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
  --output_dir "${OUT_DIR}"
  --condition "${CONDITION}"
  --image_index "${IMAGE_INDEX}"
  --resize_short "${RESIZE_SHORT}"
  --square_crop_size "${SQUARE_CROP_SIZE}"
  --viz_grid_size "${VIZ_GRID_SIZE}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu_retry "${GPU_RETRY}"
  --gpu_retry_sleep_sec "${GPU_RETRY_SLEEP_SEC}"
)

if [ "${INCLUDE_REF}" = "0" ]; then
  CMD+=(--exclude_ref)
else
  CMD+=(--include_ref)
fi

if [ "${AMP}" = "1" ]; then
  CMD+=(--amp)
fi

if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

if [ "${#}" -gt 0 ]; then
  CMD+=("$@")
fi

echo "[run_browser_anchor] OUT_DIR=${OUT_DIR}"
echo "[run_browser_anchor] HOST=${HOST} PORT=${PORT} CONDITION=${CONDITION} IMAGE_INDEX=${IMAGE_INDEX}"
echo "[run_browser_anchor] SSH port forwarding (mac/local):"
echo "  ssh -L ${PORT}:localhost:${PORT} <user>@<server>"
echo "[run_browser_anchor] command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
