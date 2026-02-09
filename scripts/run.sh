#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
EXP_ID="${EXP_ID:-ep1-SymKL}"
RUN_DIR="${ROOT_DIR}/runs/${EXP_ID}"
QUAL_DIR="${ROOT_DIR}/outputs/qual/${EXP_ID}"
RESULTS_CSV="${RUN_DIR}/results.csv"

# Visualization knobs (override via env)
# - Train-time: save selected anchor overlays periodically
DIAG_ANCHOR_EVERY="${DIAG_ANCHOR_EVERY:-1}"     # 0 disables
DIAG_ANCHOR_DIR="${DIAG_ANCHOR_DIR:-}"           # empty -> output_dir/diag_train_anchors
DIAG_ANCHOR_MAX="${DIAG_ANCHOR_MAX:-0}"         # 0 -> unlimited
DIAG_ANCHOR_PER_CALL="${DIAG_ANCHOR_PER_CALL:-1}"

# - Post-train: run diag_signal_v0.py (anchors/entropy/delta) for a few steps
RUN_POST_DIAG="${RUN_POST_DIAG:-1}"              # 1 enables
POST_DIAG_SPLIT="${POST_DIAG_SPLIT:-train}"      # train|val|test
POST_DIAG_DATA_MODE="${POST_DIAG_DATA_MODE:-train_aug}"  # train_aug|full_frame
POST_DIAG_STEPS="${POST_DIAG_STEPS:-0}"          # <=0 runs all
POST_DIAG_SAVE_ANCHORS_N="${POST_DIAG_SAVE_ANCHORS_N:-12}"
POST_DIAG_DIR="${POST_DIAG_DIR:-${RUN_DIR}/diag_signal}"

if [ ! -d "${ACDC_ROOT}" ]; then
  echo "[run.sh] ACDC_ROOT does not exist: ${ACDC_ROOT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${QUAL_DIR}"

conda run -n selo python "${ROOT_DIR}/scripts/guard_gpu.py"

conda run -n selo python "${ROOT_DIR}/train_selo_v0.py" \
		  --acdc_root "${ACDC_ROOT}" \
		  --output_dir "${RUN_DIR}" \
		  --seed 1 \
		  --epochs 1 \
		  --resize 1072 \
		  --crop_size 1072 \
		  --batch_size 4 \
		  --grad_accum_steps 4 \
		  --workers 4 \
		  --proj_warmup_steps 50 \
		  --lr_proj_after_warmup 0 \
		  --proj_type mlp \
		  --proj_mlp_hidden 256 \
		  --affinity_k 7 \
		  --affinity_tau 0.2 \
		  --affinity_anchors 256 \
		  --affinity_candidates 4096 \
		  --affinity_kcenter_top_m 1500 \
		  --affinity_per_image 1 \
		  --diag_anchor_every "${DIAG_ANCHOR_EVERY}" \
		  --diag_anchor_dir "${DIAG_ANCHOR_DIR}" \
		  --diag_anchor_max "${DIAG_ANCHOR_MAX}" \
		  --diag_anchor_per_call "${DIAG_ANCHOR_PER_CALL}" \
		  --adapter_hidden_ratio 1 \
		  --lambda_aff 1.0 \
		  --adapter_scale 1 \
		  --amp

conda run -n selo python "${ROOT_DIR}/scripts/eval_selo_v0.py" \
	  --acdc_root "${ACDC_ROOT}" \
	  --split val \
	  --resize 1080 \
	  --output_dir "${RUN_DIR}" \
	  --qual_dir "${QUAL_DIR}" \
	  --ckpt "${RUN_DIR}/adapter.pth" \
	  --exp_id "${EXP_ID}" \
	  --seed 1 \
	  --adapter_scale 1 \
	  --adapter_hidden_ratio 1 \
		  --epochs 1 \
		  --results_path "${RESULTS_CSV}"

if [ "${RUN_POST_DIAG}" = "1" ]; then
  mkdir -p "${POST_DIAG_DIR}"
  conda run -n selo python "${ROOT_DIR}/scripts/diag_signal_v0.py" \
	--acdc_root "${ACDC_ROOT}" \
	--split "${POST_DIAG_SPLIT}" \
	--data_mode "${POST_DIAG_DATA_MODE}" \
	--mode full \
	--conditions fog night rain snow \
	--resize 1080 \
	--crop_size 1072 \
	--batch_size 1 \
	--workers 0 \
	--steps "${POST_DIAG_STEPS}" \
	--save_anchors_n "${POST_DIAG_SAVE_ANCHORS_N}" \
	--ckpt "${RUN_DIR}/adapter.pth" \
	--affinity_k 7 \
	--affinity_tau 0.07 \
	--affinity_anchors 256 \
	--affinity_candidates 4096 \
	--affinity_kcenter_top_m 1500 \
	--affinity_per_image 1 \
	--adapter_hidden_ratio 1 \
	--adapter_scale 1 \
	--proj_type auto \
	--output_dir "${POST_DIAG_DIR}"
fi
