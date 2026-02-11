#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/exp/dino_segformer_layer_match/run_dino_segformer_layer_match.py"

ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/exp/dino_segformer_layer_match/full_all_ref_bs6_1072_viz67_watch}"
MON_INTERVAL_SEC="${MON_INTERVAL_SEC:-10}"
BATCH_SIZE="${BATCH_SIZE:-6}"
WORKERS="${WORKERS:-2}"
INCLUDE_REF="${INCLUDE_REF:-1}"
CONDITIONS="${CONDITIONS:-fog night rain snow}"
CKA_SSM_GRID="${CKA_SSM_GRID:-14}"
VIZ_GRID_SIZE="${VIZ_GRID_SIZE:-0}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-200}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1000}"
VIZ_EVERY="${VIZ_EVERY:-200}"
VIZ_MAX="${VIZ_MAX:-0}"
VIZ_ANCHOR_MODE="${VIZ_ANCHOR_MODE:-center}"
GPU_GUARD_RETRY="${GPU_GUARD_RETRY:-5}"
GPU_GUARD_RETRY_SLEEP_SEC="${GPU_GUARD_RETRY_SLEEP_SEC:-2}"
CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  fi
fi

mkdir -p "${OUT_DIR}"
MON_LOG="${OUT_DIR}/monitor.log"
RUN_LOG="${OUT_DIR}/run.console.log"
EVAL_LOG="${OUT_DIR}/eval.log"
RUN_PID=""

on_signal() {
  local sig="$1"
  local ts
  ts="$(date '+%F %T')"
  echo "${ts} | watch got signal=${sig}" | tee -a "${MON_LOG}"
  if [ "${sig}" = "HUP" ]; then
    echo "${ts} | HUP ignored; keep child alive pid=${RUN_PID}" | tee -a "${MON_LOG}"
    return 0
  fi
  if [ -n "${RUN_PID}" ] && kill -0 "${RUN_PID}" 2>/dev/null; then
    echo "${ts} | forwarding SIGTERM to child pid=${RUN_PID}" | tee -a "${MON_LOG}"
    kill -TERM "${RUN_PID}" 2>/dev/null || true
  fi
  exit 128
}

trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

{
  echo "===== watch_and_run start: $(date '+%F %T') ====="
  echo "ACDC_ROOT=${ACDC_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "MON_INTERVAL_SEC=${MON_INTERVAL_SEC}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "WORKERS=${WORKERS}"
  echo "INCLUDE_REF=${INCLUDE_REF}"
  echo "CONDITIONS=${CONDITIONS}"
  echo "CKA_SSM_GRID=${CKA_SSM_GRID}"
  echo "VIZ_GRID_SIZE=${VIZ_GRID_SIZE}"
  echo "PROGRESS_LOG_EVERY=${PROGRESS_LOG_EVERY}"
  echo "SNAPSHOT_EVERY=${SNAPSHOT_EVERY}"
  echo "VIZ_EVERY=${VIZ_EVERY}"
  echo "VIZ_MAX=${VIZ_MAX}"
  echo "VIZ_ANCHOR_MODE=${VIZ_ANCHOR_MODE}"
  echo "GPU_GUARD_RETRY=${GPU_GUARD_RETRY}"
  echo "GPU_GUARD_RETRY_SLEEP_SEC=${GPU_GUARD_RETRY_SLEEP_SEC}"
  echo "CONDA_BIN=${CONDA_BIN}"
} | tee -a "${MON_LOG}"

if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
  echo "[watch] conda binary not found. set CONDA_BIN or fix PATH." | tee -a "${MON_LOG}"
  exit 1
fi

GPU_GUARD_OK=0
for ((i=1; i<=GPU_GUARD_RETRY; i++)); do
  "${CONDA_BIN}" run -n selo python "${ROOT_DIR}/scripts/guard_gpu.py" >> "${MON_LOG}" 2>&1
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    GPU_GUARD_OK=1
    echo "[watch] GPU guard passed on try=${i}/${GPU_GUARD_RETRY}" | tee -a "${MON_LOG}"
    break
  fi
  echo "[watch] GPU guard failed on try=${i}/${GPU_GUARD_RETRY} (rc=${rc})" | tee -a "${MON_LOG}"
  if [ "${i}" -lt "${GPU_GUARD_RETRY}" ]; then
    sleep "${GPU_GUARD_RETRY_SLEEP_SEC}"
  fi
done
if [ "${GPU_GUARD_OK}" -ne 1 ]; then
  echo "[watch] GPU guard failed after retries. stop." | tee -a "${MON_LOG}"
  exit 1
fi

CMD=(
  "${CONDA_BIN}" run -n selo python "${SCRIPT_PATH}"
  --acdc_root "${ACDC_ROOT}"
  --output_dir "${OUT_DIR}"
  --resize_short 1072
  --square_crop_size 1072
  --grid_size "${CKA_SSM_GRID}"
  --viz_grid_size "${VIZ_GRID_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --progress_log_every "${PROGRESS_LOG_EVERY}"
  --snapshot_every "${SNAPSHOT_EVERY}"
  --viz_every "${VIZ_EVERY}"
  --viz_max "${VIZ_MAX}"
  --viz_anchor_mode "${VIZ_ANCHOR_MODE}"
  --anchor_x 0.5
  --anchor_y 0.5
  --amp
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

{
  echo "[watch] run command:"
  printf ' %q' "${CMD[@]}"
  echo
} | tee -a "${MON_LOG}"

# Keep full stdout/stderr for crash diagnosis.
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}" > "${RUN_LOG}" 2>&1 &
else
  "${CMD[@]}" > "${RUN_LOG}" 2>&1 &
fi
RUN_PID=$!
echo "[watch] spawned pid=${RUN_PID}" | tee -a "${MON_LOG}"

while kill -0 "${RUN_PID}" 2>/dev/null; do
  TS="$(date '+%F %T')"
  GPU_LINE="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  echo "${TS} | alive pid=${RUN_PID} | gpu(mem_used,mem_total,util,temp)=${GPU_LINE}" >> "${MON_LOG}"
  if [ -f "${EVAL_LOG}" ]; then
    tail -n 3 "${EVAL_LOG}" | sed "s/^/${TS} | eval | /" >> "${MON_LOG}"
  fi
  sleep "${MON_INTERVAL_SEC}"
done

set +e
wait "${RUN_PID}"
EXIT_CODE=$?
set -e

TS_END="$(date '+%F %T')"
echo "${TS_END} | process exited | code=${EXIT_CODE}" | tee -a "${MON_LOG}"

echo "${TS_END} | last run.console.log lines" >> "${MON_LOG}"
tail -n 80 "${RUN_LOG}" >> "${MON_LOG}" 2>/dev/null || true

echo "${TS_END} | error keyword scan" >> "${MON_LOG}"
if command -v rg >/dev/null 2>&1; then
  rg -n -i "out of memory|cuda.*memory|cudnn_status_alloc_failed|killed|oom|runtimeerror|traceback" "${RUN_LOG}" >> "${MON_LOG}" 2>/dev/null || true
else
  grep -Eni "out of memory|cuda.*memory|cudnn_status_alloc_failed|killed|oom|runtimeerror|traceback" "${RUN_LOG}" >> "${MON_LOG}" 2>/dev/null || true
fi

echo "${TS_END} | dmesg OOM/Xid scan" >> "${MON_LOG}"
dmesg -T 2>&1 | grep -Ei "out of memory|oom|killed process|NVRM|Xid" | tail -n 50 >> "${MON_LOG}" || true

echo "===== watch_and_run end: ${TS_END} =====" | tee -a "${MON_LOG}"
exit "${EXIT_CODE}"
