#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TASK_OVERRIDE="${TASK_OVERRIDE:-task=libero_uncond_2cam224_1e-4}"
if (($# > 0)) && [[ "$1" == task=* ]]; then
  TASK_OVERRIDE="$1"
  shift
fi

MODES=(
  fastwam
  joint
  early_matched
  sandwich_dense
  late_matched
  clcf
  clcf_wo_causal
  clcf_wo_temporal_local
  clcf_wo_spatial_c2f
)

for mode in "${MODES[@]}"; do
  dropout="${FUTURE_MASK_DROPOUT:-0.3}"
  if [[ "${mode}" == "fastwam" || "${mode}" == "joint" ]]; then
    dropout="0.0"
  fi

  RUN_ID="${RUN_ID_PREFIX:-clcf}_${mode}" \
  bash scripts/train_zero2.sh "${NPROC_PER_NODE}" \
    "${TASK_OVERRIDE}" \
    "model=fastwam_3dmask" \
    "model.visibility.mode=${mode}" \
    "model.visibility.future_mask_dropout=${dropout}" \
    "wandb.name=${mode}" \
    "$@"
done
