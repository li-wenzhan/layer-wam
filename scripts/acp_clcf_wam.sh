#!/bin/bash
# ACP entrypoint for CLCF-WAM / LayerWAM experiments.
#
# This script is intended to be submitted as the command script of an ACP job.
# It mirrors the ACP flow from tmp/acp_for_merl.sh:
#   1. clean stale runtime state
#   2. define cluster-specific paths and experiment knobs
#   3. inject a path-based conda environment
#   4. configure CUDA/NCCL/cache/offline env vars
#   5. run one experiment command and tee logs
#   6. scan logs, clean helpers, and exit with a meaningful code

set +e

# ==========================================
# 0. User-editable ACP config
# ==========================================

# TODO: Change to the layer-wam project directory on the ACP filesystem.
# Example: /mnt/afs/TODO_USER/layer-wam
PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam}"

# TODO: Change to the absolute path of the conda env directory.
# Example: /mnt/afs/TODO_USER/envs/fastwam
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/fastwam}"

# TODO: Change to the directory that stores Wan / FastWAM / ActionDiT checkpoints.
# Example: /mnt/afs/TODO_USER/checkpoints
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam/checkpoints}"

# TODO: Change to a persistent cache directory visible inside ACP jobs.
# Example: /mnt/afs/TODO_USER/cache
CACHE_ROOT="${CACHE_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam/cache}"

# TODO: Change to a persistent log root. Logs from this script are written below it.
# Example: /mnt/afs/TODO_USER/tmp/acp_logs
LOG_ROOT="${LOG_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam/tmp/acp_logs}"

# TODO: Select what this ACP job should run.
# Options:
#   precompute_text  - precompute T5 context caches
#   debug_mask       - inspect the layer-wise visibility mask
#   train            - train one visibility mode
#   ablation         - run all first-version modes sequentially
#   eval_libero      - evaluate one checkpoint on LIBERO
#   eval_robotwin    - evaluate one checkpoint on RoboTwin
RUN_KIND="${RUN_KIND:-precompute_text}"

# TODO: Choose TASK_NAME according to RUN_KIND.
# Available task configs live in configs/task/*.yaml; pass the file stem without ".yaml".
#
# RUN_KIND=precompute_text:
#   Use the same task as the training/evaluation job whose text cache you want to build.
#   Recommended CLCF-WAM tasks:
#     libero_uncond_2cam224_1e-4
#     robotwin_uncond_3cam_384_1e-4
#
# RUN_KIND=train:
#   Use a trainable task config, then override model=fastwam_3dmask below.
#   Recommended CLCF-WAM tasks:
#     libero_uncond_2cam224_1e-4
#     robotwin_uncond_3cam_384_1e-4
#   Original Fast-WAM variants also exist for reproduction/debug:
#     libero_joint_2cam224_1e-4
#     libero_idm_2cam224_1e-4
#     robotwin_joint_3cam_384_1e-4
#     robotwin_idm_3cam_384_1e-4
#
# RUN_KIND=ablation:
#   Use the dataset/task family once; the script loops over visibility modes.
#   Recommended:
#     libero_uncond_2cam224_1e-4
#     robotwin_uncond_3cam_384_1e-4
#
# RUN_KIND=eval_libero:
#   Must use a LIBERO task:
#     libero_uncond_2cam224_1e-4
#     libero_joint_2cam224_1e-4
#     libero_idm_2cam224_1e-4
#
# RUN_KIND=eval_robotwin:
#   Must use a RoboTwin task:
#     robotwin_uncond_3cam_384_1e-4
#     robotwin_joint_3cam_384_1e-4
#     robotwin_idm_3cam_384_1e-4
#
# RUN_KIND=debug_mask:
#   TASK_NAME is only used in the log directory name; visibility behavior is controlled by VISIBILITY_MODE.
TASK_NAME="${TASK_NAME:-robotwin_uncond_3cam_384_1e-4}"

# TODO: Change GPU list according to the ACP resource allocation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# TODO: Model / visibility knobs for RUN_KIND=train, ablation, debug_mask.
#
# MODEL_CONFIG is the Hydra model config name under configs/model/*.yaml.
# Use the file stem without ".yaml".
#
# Recommended for our CLCF-WAM / LayerWAM experiments:
#   MODEL_CONFIG=fastwam_3dmask
#     Enables model.visibility.mode and model.visibility.future_mask_dropout.
#     This is the only MODEL_CONFIG that should be paired with VISIBILITY_MODE.
#
# Original Fast-WAM reproduction/debug configs:
#   MODEL_CONFIG=fastwam
#     Original Fast-WAM no-future action path. VISIBILITY_MODE is ignored in train.
#   MODEL_CONFIG=fastwam_joint
#     Original all-layer joint future-video reference. VISIBILITY_MODE is ignored in train.
#   MODEL_CONFIG=fastwam_idm
#     Original IDM variant. VISIBILITY_MODE is ignored in train.
#
# RUN_KIND-specific behavior:
#   precompute_text:
#     MODEL_CONFIG/VISIBILITY_MODE do not affect the text cache; only TASK_NAME matters.
#   debug_mask:
#     MODEL_CONFIG is ignored; VISIBILITY_MODE selects which visibility policy to inspect.
#   train:
#     If MODEL_CONFIG=fastwam_3dmask, this script passes:
#       model=fastwam_3dmask
#       model.visibility.mode=${VISIBILITY_MODE}
#       model.visibility.future_mask_dropout=${FUTURE_MASK_DROPOUT or auto-resolved value}
#     If MODEL_CONFIG is an original Fast-WAM config, this script passes only model=${MODEL_CONFIG}.
#   ablation:
#     MODEL_CONFIG/VISIBILITY_MODE are ignored; scripts/run_clcf_ablation.sh always loops over
#     all first-version visibility modes using model=fastwam_3dmask.
#
# VISIBILITY_MODE is valid only for MODEL_CONFIG=fastwam_3dmask or RUN_KIND=debug_mask.
# Supported VISIBILITY_MODE values:
#   fastwam
#     Original no-future direct action-to-video visibility; auto dropout = 0.0.
#   joint
#     All layers can directly read all future video tokens; auto dropout = 0.0.
#   early_matched
#     P0+P1 all-future dense, P2+P3 closed; 40% visible-layer budget.
#   sandwich_dense
#     P1+P2 all-future dense, P0+P3 closed; 40% visible-layer budget.
#   late_matched
#     P3 all-future dense, P0+P1+P2 closed; 40% visible-layer budget.
#   clcf
#     P1 coarse direct-edge causal sparse, P2 local direct-edge causal dense, P0/P3 closed.
#   clcf_wo_causal
#     Removes direct-edge causal constraint while keeping middle-layer and spatial schedule.
#   clcf_wo_temporal_local
#     Replaces P2 local window with causal prefix; tests temporal locality.
#   clcf_wo_spatial_c2f
#     Makes P1 dense instead of sparse; tests spatial coarse-to-fine.
#
# Concrete examples:
#   Train CLCF:
#     MODEL_CONFIG=fastwam_3dmask VISIBILITY_MODE=clcf RUN_KIND=train bash scripts/acp_clcf_wam.sh
#   Train budget-matched sandwich:
#     MODEL_CONFIG=fastwam_3dmask VISIBILITY_MODE=sandwich_dense RUN_KIND=train bash scripts/acp_clcf_wam.sh
#   Reproduce original Fast-WAM baseline:
#     MODEL_CONFIG=fastwam RUN_KIND=train bash scripts/acp_clcf_wam.sh
#   Inspect CLCF mask only:
#     RUN_KIND=debug_mask VISIBILITY_MODE=clcf bash scripts/acp_clcf_wam.sh
MODEL_CONFIG="${MODEL_CONFIG:-fastwam_3dmask}"
VISIBILITY_MODE="${VISIBILITY_MODE:-clcf}"

# Use "auto" to follow our convention:
#   fastwam/joint -> 0.0
#   scheduled variants -> 0.3
FUTURE_MASK_DROPOUT="${FUTURE_MASK_DROPOUT:-auto}"

# TODO: Tune if 4 * H100 is not enough or if a different global batch is desired.
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-false}"

# Optional Hydra overrides. Keep this simple: space-separated key=value entries.
# TODO: Add project-specific overrides here if needed.
HYDRA_EXTRA_ARGS="${HYDRA_EXTRA_ARGS:-}"

# Text-cache knobs.
OVERWRITE_TEXT_CACHE="${OVERWRITE_TEXT_CACHE:-true}"

# Evaluation knobs.
# TODO: Required for RUN_KIND=eval_libero or eval_robotwin.
# Example: /mnt/afs/TODO_USER/checkpoints/TODO_STEP.pt
CKPT_PATH="${CKPT_PATH:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam/checkpoints/TODO_STEP.pt}"

# TODO: Optional but recommended for evaluation. Set to "none" to omit.
# Example: /mnt/afs/TODO_USER/dataset_stats.json
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/mnt/afs/task3_2/L202500276_lwz/projects/layer-wam/data/dataset_stats.json}"

# X11/MuJoCo helpers. Training does not need Xvfb; simulator evaluation usually does.
USE_XVFB="${USE_XVFB:-false}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-false}"

# Offline mode is recommended when all models and datasets have already been cached.
HF_OFFLINE="${HF_OFFLINE:-1}"

# ==========================================
# 1. Helpers
# ==========================================

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

contains_todo() {
  [[ "$1" == *"TODO"* ]]
}

require_no_todo() {
  local name="$1"
  local value="$2"
  if contains_todo "$value"; then
    echo "ERROR: ${name} still contains TODO placeholder: ${value}"
    echo "Please edit scripts/acp_clcf_wam.sh or pass ${name}=... as an ACP environment variable."
    return 1
  fi
  return 0
}

resolve_dropout() {
  if [[ "${FUTURE_MASK_DROPOUT}" != "auto" ]]; then
    echo "${FUTURE_MASK_DROPOUT}"
    return 0
  fi

  if [[ "${VISIBILITY_MODE}" == "fastwam" || "${VISIBILITY_MODE}" == "joint" ]]; then
    echo "0.0"
  else
    echo "0.3"
  fi
}

parse_extra_args() {
  EXTRA_ARGS=()
  if [[ -n "${HYDRA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${HYDRA_EXTRA_ARGS})
  fi
}

run_and_log() {
  echo "========== COMMAND =========="
  printf '%q ' "$@"
  echo
  echo "============================="
  "$@" 2>&1 | tee "${LOG_FILE}"
  return "${PIPESTATUS[0]}"
}

scan_log_for_errors() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    return 0
  fi
  grep -qE \
    "Traceback|RuntimeError|ImportError|ModuleNotFoundError|CUDA out of memory|NCCL error|Error executing job|mujoco.FatalError|gladLoadGL error|Failed to open display" \
    "${LOG_FILE}"
}

# ==========================================
# 2. Cleanup
# ==========================================

echo "========== CLEANUP =========="
if [[ "${USE_XVFB}" == "true" ]]; then
  pkill -f Xvfb 2>/dev/null || true
  rm -f /tmp/.X99-lock
fi

# This repo does not use Ray for training, but stopping stale Ray state is harmless
# inside an isolated ACP container and keeps simulator jobs cleaner.
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

# ==========================================
# 3. Validate user-editable paths
# ==========================================

require_no_todo PROJECT_DIR "${PROJECT_DIR}" || exit 2
require_no_todo CONDA_ENV_DIR "${CONDA_ENV_DIR}" || exit 2
require_no_todo DIFFSYNTH_MODEL_BASE_PATH "${DIFFSYNTH_MODEL_BASE_PATH}" || exit 2
require_no_todo CACHE_ROOT "${CACHE_ROOT}" || exit 2
require_no_todo LOG_ROOT "${LOG_ROOT}" || exit 2

if [[ "${RUN_KIND}" == "eval_libero" || "${RUN_KIND}" == "eval_robotwin" ]]; then
  require_no_todo CKPT_PATH "${CKPT_PATH}" || exit 2
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 2
fi

if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
  echo "ERROR: Cannot find python in CONDA_ENV_DIR: ${CONDA_ENV_DIR}/bin/python"
  exit 2
fi

# ==========================================
# 4. Environment setup
# ==========================================

echo "========== INIT ENVIRONMENT =========="
cd "${PROJECT_DIR}" || exit 2

DROPOUT="$(resolve_dropout)"
RUN_STAMP="$(timestamp)"
RUN_LABEL="${RUN_KIND}_${TASK_NAME}_${VISIBILITY_MODE}_${RUN_STAMP}"
LOG_DIR="${LOG_ROOT}/${RUN_LABEL}"
LOG_FILE="${LOG_DIR}/console.log"
mkdir -p "${LOG_DIR}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CONDA_ENV_DIR=${CONDA_ENV_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_KIND=${RUN_KIND}"
echo "TASK_NAME=${TASK_NAME}"
echo "VISIBILITY_MODE=${VISIBILITY_MODE}"
echo "FUTURE_MASK_DROPOUT=${DROPOUT}"

echo "Injecting conda env to PATH: ${CONDA_ENV_DIR}/bin"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

export HF_DATASETS_OFFLINE="${HF_OFFLINE}"
export TRANSFORMERS_OFFLINE="${HF_OFFLINE}"
export HF_HUB_OFFLINE="${HF_OFFLINE}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,eth0,bond0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${USE_XVFB}" == "true" ]]; then
  echo "========== INIT XVFB =========="
  if [[ "${INSTALL_SYSTEM_DEPS}" == "true" ]]; then
    echo "Installing X11 / GL dependencies. TODO: disable INSTALL_SYSTEM_DEPS if ACP image already contains them."
    export DEBIAN_FRONTEND=noninteractive
    if [[ "$(id -u)" == "0" ]]; then
      apt-get update -yq
      apt-get install -yq \
        xvfb x11-utils mesa-utils \
        libgl1 libglx0 libglx-mesa0 libgl1-mesa-glx libgl1-mesa-dev \
        libglib2.0-0 libglfw3
    else
      echo "WARNING: INSTALL_SYSTEM_DEPS=true but current user is not root; skipping apt-get."
    fi
  fi
  Xvfb :99 -screen 0 1024x768x24 &
  XVFB_PID=$!
  export DISPLAY=:99
  export MUJOCO_GL="${MUJOCO_GL:-glx}"
fi

parse_extra_args

# ==========================================
# 5. Build command
# ==========================================

CMD=()
case "${RUN_KIND}" in
  precompute_text)
    CMD=(
      python scripts/precompute_text_embeds.py
      "task=${TASK_NAME}"
      "+overwrite=${OVERWRITE_TEXT_CACHE}"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  debug_mask)
    CMD=(
      python scripts/debug_visibility_mask.py
      --mode "${VISIBILITY_MODE}"
      --future-mask-dropout "${DROPOUT}"
    )
    ;;

  train)
    CMD=(
      bash scripts/train_zero2.sh "${NPROC_PER_NODE}"
      "task=${TASK_NAME}"
      "model=${MODEL_CONFIG}"
      "batch_size=${BATCH_SIZE}"
      "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
      "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}"
      "wandb.name=${MODEL_CONFIG}_${VISIBILITY_MODE}"
    )
    if [[ "${MODEL_CONFIG}" == "fastwam_3dmask" ]]; then
      CMD+=(
        "model.visibility.mode=${VISIBILITY_MODE}"
        "model.visibility.future_mask_dropout=${DROPOUT}"
      )
    fi
    CMD+=("${EXTRA_ARGS[@]}")
    ;;

  ablation)
    export NPROC_PER_NODE
    export FUTURE_MASK_DROPOUT="${DROPOUT}"
    export RUN_ID_PREFIX="${RUN_ID_PREFIX:-acp_clcf}"
    CMD=(
      bash scripts/run_clcf_ablation.sh
      "task=${TASK_NAME}"
      "batch_size=${BATCH_SIZE}"
      "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
      "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  eval_libero)
    CMD=(
      python experiments/libero/run_libero_manager.py
      "task=${TASK_NAME}"
      "ckpt=${CKPT_PATH}"
      "MULTIRUN.num_gpus=${NPROC_PER_NODE}"
      "${EXTRA_ARGS[@]}"
    )
    if [[ "${DATASET_STATS_PATH}" != "none" && ! "${DATASET_STATS_PATH}" == *"TODO"* ]]; then
      CMD+=("EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}")
    fi
    ;;

  eval_robotwin)
    CMD=(
      python experiments/robotwin/run_robotwin_manager.py
      "task=${TASK_NAME}"
      "ckpt=${CKPT_PATH}"
      "MULTIRUN.num_gpus=${NPROC_PER_NODE}"
      "${EXTRA_ARGS[@]}"
    )
    if [[ "${DATASET_STATS_PATH}" != "none" && ! "${DATASET_STATS_PATH}" == *"TODO"* ]]; then
      CMD+=("EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}")
    fi
    ;;

  *)
    echo "ERROR: Unsupported RUN_KIND=${RUN_KIND}"
    echo "Expected one of: precompute_text, debug_mask, train, ablation, eval_libero, eval_robotwin"
    exit 2
    ;;
esac

# ==========================================
# 6. Run and summarize
# ==========================================

echo "========== START JOB =========="
run_and_log "${CMD[@]}"
EXIT_CODE=$?

if scan_log_for_errors; then
  echo "!! ERROR DETECTED: serious failure pattern found in ${LOG_FILE}; forcing EXIT_CODE=1"
  EXIT_CODE=1
fi

if [[ -n "${XVFB_PID:-}" ]]; then
  kill "${XVFB_PID}" 2>/dev/null || true
fi

echo "========== JOB FINISHED =========="
echo "EXIT_CODE=${EXIT_CODE}"
echo "LOG_FILE=${LOG_FILE}"

exit "${EXIT_CODE}"
