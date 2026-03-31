#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_ENV_NAME="${BASE_ENV_NAME:-neural_gaffer_5090}"
ENV_PREFIX="${1:-/4T/conda_envs/dilightnet_baseline}"
DILIGHTNET_REPO="${DILIGHTNET_REPO:-$REPO_ROOT/external/DiLightNet_full}"
SETUP_MODE="${SETUP_MODE:-clone}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 1
fi

if [[ ! -d "$DILIGHTNET_REPO" ]]; then
  echo "DiLightNet repo not found at $DILIGHTNET_REPO"
  exit 1
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  if [[ "$SETUP_MODE" == "official" ]]; then
    echo "[setup] creating official-style env at $ENV_PREFIX"
    conda create -p "$ENV_PREFIX" \
      python=3.10 \
      pytorch==2.5.1 \
      torchvision==0.20.1 \
      pytorch-cuda==12.4 \
      mkl==2023.1.0 \
      -c pytorch -c nvidia -y
  else
    echo "[setup] cloning $BASE_ENV_NAME -> $ENV_PREFIX"
    conda create --clone "$BASE_ENV_NAME" -p "$ENV_PREFIX" -y
  fi
else
  echo "[setup] reusing existing env at $ENV_PREFIX"
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
PIP_BIN="$ENV_PREFIX/bin/pip"

echo "[setup] python: $("$PYTHON_BIN" --version)"
echo "[setup] upgrading pip tooling"
"$PIP_BIN" install --upgrade pip setuptools wheel

echo "[setup] installing official DiLightNet requirements"
"$PIP_BIN" install -r "$DILIGHTNET_REPO/requirements.txt"

echo "[setup] smoke test imports"
"$PYTHON_BIN" -c "import simple_parsing, trimesh, bpy, bpy_helper, dust3r; print('DiLightNet import smoke test passed')"

echo "[setup] ready: $ENV_PREFIX"
