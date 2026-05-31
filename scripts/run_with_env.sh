#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/itdd-pfs/wangyuhao/long_horizon/verl-agent}
CONDA_ROOT=${CONDA_ROOT:-/itdd-pfs/wangyuhao/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-verl-agent}
WANDB_KEY_FILE=${WANDB_KEY_FILE:-/itdd-pfs/wangyuhao/.wandb_api_key}

usage() {
    cat >&2 <<'EOF'
Usage:
  CONDA_ENV_NAME=<env> scripts/run_with_env.sh <script> [args...]

Common env overrides:
  REPO_ROOT=/path/to/verl-agent
  CONDA_ROOT=/path/to/miniconda3
  WANDB_API_KEY=<key>
  WANDB_KEY_FILE=/path/to/key_file
  WANDB_MODE=online|offline|disabled
EOF
}

if [ "$#" -lt 1 ]; then
    usage
    exit 2
fi

TARGET_SCRIPT=$1
shift

cd "$REPO_ROOT"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
elif [ -x "$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin/python3" ]; then
    export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:$PATH"
else
    echo "Cannot find conda environment '$CONDA_ENV_NAME' under '$CONDA_ROOT'." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/itdd-pfs/wangyuhao/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export WANDB_DIR=${WANDB_DIR:-/itdd-pfs/wangyuhao/logs/wandb}
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_KEY_FILE" ]; then
    WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")
    export WANDB_API_KEY
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ "$WANDB_MODE" = "online" ]; then
    echo "WANDB_API_KEY is not set. Export it or put it in '$WANDB_KEY_FILE'." >&2
    exit 1
fi

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Script not found: $TARGET_SCRIPT" >&2
    exit 1
fi

echo "REPO_ROOT=$REPO_ROOT"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "WANDB_MODE=$WANDB_MODE"
echo "TARGET_SCRIPT=$TARGET_SCRIPT"

exec bash "$TARGET_SCRIPT" "$@"
