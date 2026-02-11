#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/exp/dino_segformer_layer_match/run_dino_segformer_layer_match.py"

ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/exp/dino_segformer_layer_match/night_interactive}"
CONDITIONS="${CONDITIONS:-night}"
INCLUDE_REF="${INCLUDE_REF:-0}"

RESIZE_SHORT="${RESIZE_SHORT:-1072}"
SQUARE_CROP_SIZE="${SQUARE_CROP_SIZE:-1072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-0}"
CKA_SSM_GRID="${CKA_SSM_GRID:-67}"
VIZ_GRID_SIZE="${VIZ_GRID_SIZE:-67}"
VIZ_ANCHOR_MODE="${VIZ_ANCHOR_MODE:-quadrant4}"
VIZ_EVERY="${VIZ_EVERY:-1}"
VIZ_MAX="${VIZ_MAX:-0}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-20}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-50}"
MAX_IMAGES="${MAX_IMAGES:-0}"
AMP="${AMP:-1}"
GPU_CHECK_RETRY="${GPU_CHECK_RETRY:-8}"
GPU_CHECK_RETRY_SLEEP_SEC="${GPU_CHECK_RETRY_SLEEP_SEC:-2}"

CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  fi
fi

if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
  echo "[interactive] conda binary not found." >&2
  echo "확인 커맨드:" >&2
  echo "  nvidia-smi" >&2
  echo "  python -c \"import torch; print(torch.cuda.is_available())\"" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
CONSOLE_LOG="${OUT_DIR}/interactive.console.log"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[GPU 체크 실패] nvidia-smi 실행 실패" >&2
  echo "확인 커맨드:" >&2
  echo "  nvidia-smi" >&2
  echo "  python -c \"import torch; print(torch.cuda.is_available())\"" >&2
  exit 1
fi

GPU_OK=0
for ((i=1; i<=GPU_CHECK_RETRY; i++)); do
  if "${CONDA_BIN}" run --no-capture-output -n selo python -c "import sys, torch; ok=torch.cuda.is_available(); print(f'[interactive] torch.cuda.is_available={ok}'); sys.exit(0 if ok else 1)"; then
    GPU_OK=1
    echo "[interactive] GPU check passed on try=${i}/${GPU_CHECK_RETRY}" | tee -a "${CONSOLE_LOG}"
    break
  fi
  echo "[interactive] GPU check failed on try=${i}/${GPU_CHECK_RETRY}" | tee -a "${CONSOLE_LOG}"
  if [ "${i}" -lt "${GPU_CHECK_RETRY}" ]; then
    sleep "${GPU_CHECK_RETRY_SLEEP_SEC}"
  fi
done
if [ "${GPU_OK}" -ne 1 ]; then
  echo "[GPU 체크 실패] torch.cuda.is_available() == False" >&2
  echo "확인 커맨드:" >&2
  echo "  nvidia-smi" >&2
  echo "  python -c \"import torch; print(torch.cuda.is_available())\"" >&2
  exit 1
fi

CMD=(
  "${CONDA_BIN}" run --no-capture-output -n selo python -u "${SCRIPT_PATH}"
  --acdc_root "${ACDC_ROOT}"
  --output_dir "${OUT_DIR}"
  --resize_short "${RESIZE_SHORT}"
  --square_crop_size "${SQUARE_CROP_SIZE}"
  --grid_size "${CKA_SSM_GRID}"
  --viz_grid_size "${VIZ_GRID_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --viz_anchor_mode "${VIZ_ANCHOR_MODE}"
  --viz_every "${VIZ_EVERY}"
  --viz_max "${VIZ_MAX}"
  --progress_log_every "${PROGRESS_LOG_EVERY}"
  --snapshot_every "${SNAPSHOT_EVERY}"
)

read -r -a COND_ARR <<< "${CONDITIONS}"
if [ "${#COND_ARR[@]}" -gt 0 ]; then
  CMD+=(--conditions "${COND_ARR[@]}")
fi

if [ "${INCLUDE_REF}" = "0" ]; then
  CMD+=(--exclude_ref)
else
  CMD+=(--include_ref)
fi

if [ "${MAX_IMAGES}" != "0" ]; then
  CMD+=(--max_images "${MAX_IMAGES}")
fi

if [ "${AMP}" = "1" ]; then
  CMD+=(--amp)
fi

{
  echo "===== interactive run start: $(date '+%F %T') ====="
  echo "OUT_DIR=${OUT_DIR}"
  echo "CONDITIONS=${CONDITIONS}"
  echo "INCLUDE_REF=${INCLUDE_REF}"
  echo "BATCH_SIZE=${BATCH_SIZE}, WORKERS=${WORKERS}"
  echo "CKA_SSM_GRID=${CKA_SSM_GRID}, VIZ_GRID_SIZE=${VIZ_GRID_SIZE}, VIZ_ANCHOR_MODE=${VIZ_ANCHOR_MODE}"
  echo "VIZ_EVERY=${VIZ_EVERY}, PROGRESS_LOG_EVERY=${PROGRESS_LOG_EVERY}, SNAPSHOT_EVERY=${SNAPSHOT_EVERY}"
  echo "GPU_CHECK_RETRY=${GPU_CHECK_RETRY}, GPU_CHECK_RETRY_SLEEP_SEC=${GPU_CHECK_RETRY_SLEEP_SEC}"
  echo "[interactive] run command:"
  printf ' %q' "${CMD[@]}"
  echo
} | tee -a "${CONSOLE_LOG}"

set +e
"${CMD[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"
RC=${PIPESTATUS[0]}
set -e

echo "===== interactive run end: $(date '+%F %T') | rc=${RC} =====" | tee -a "${CONSOLE_LOG}"
exit "${RC}"
