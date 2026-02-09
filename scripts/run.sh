#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACDC_ROOT="${ACDC_ROOT:-/mnt/d/ACDC}"
EXP_ID="full-affinity"
RUN_DIR="${ROOT_DIR}/runs/${EXP_ID}"
QUAL_DIR="${ROOT_DIR}/outputs/qual/${EXP_ID}"

mkdir -p "${RUN_DIR}" "${QUAL_DIR}"

conda run -n selo python "${ROOT_DIR}/scripts/guard_gpu.py"

conda run -n selo python "${ROOT_DIR}/train_selo_v0.py" \
	  --acdc_root "${ACDC_ROOT}" \
	  --output_dir "${RUN_DIR}" \
	  --seed 1 \
	  --epochs 5 \
	  --resize 1072 \
	  --crop_size 1072 \
	  --batch_size 4 \
	  --grad_accum_steps 4 \
	  --workers 4 \
	  --proj_warmup_steps 100 \
	  --affinity_k 67 \
	  --affinity_tau 0.07 \
	  --affinity_anchors 128 \
	  --affinity_candidates 4096 \
	  --affinity_kcenter_top_m 1500 \
	  --affinity_per_image 1 \
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
  --epochs 5
