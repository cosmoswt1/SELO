#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/train_stage3_cka.py"

ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
RUN_ID="${RUN_ID:=v1}"
OUT_DIR="${OUT_DIR:=${ROOT_DIR}/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/${RUN_ID}}"

CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  fi
fi

if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
  echo "[run_train] conda binary not found." >&2
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
  # data / schedule
  --resize 1072
  --crop_size 1072
  --epochs 5
  --batch_size 1
  --grad_accum_steps 4
  --workers 2
  --lr 1e-4
  --weight_decay 1e-2
  --max_grad_norm 1.0
  # model / cka sampling
  --adapter_bottleneck 128
  --gate_bias_init -4.0
  --force_gate_one 1
  --gate_detach_align 1
  --local_window_size 16
  --local_windows_total 10
  --local_windows_per_step 10
  --boundary_ratio_local 0.6
  # trust-region
  --delta_out 0.02
  --delta_upd 0.5
  --use_upd_loss 0
  --lambda_out_init 5.0
  --lambda_upd_init 0.0
  --dual_lr_out 0.0
  --dual_lr_upd 0.0
  --lambda_max 10.0
  --anchor_conf_gamma 2.0
  --anchor_conf_thresh 0.93
  --anchor_temperature 1.0
  # eval / diagnostics
  --eval_every_epoch 1
  --eval_split val
  --eval_resize 1080
  --eval_batch_size 1
  --eval_workers 0
  --diag_heavy_interval 50
  --diag_den_warn 1e-4
  --diag_den_critical 1e-6
  --amp
)

if [ "${#}" -gt 0 ]; then
  CMD+=("$@")
fi

echo "[run_train] OUT_DIR=${OUT_DIR}"
echo "[run_train] command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
