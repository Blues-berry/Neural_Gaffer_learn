#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${DILIGHTNET_ENV_PREFIX:-/4T/conda_envs/dilightnet_baseline}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
DILIGHTNET_REPO="${DILIGHTNET_REPO:-$REPO_ROOT/external/DiLightNet_full}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing baseline env at $ENV_PREFIX"
  echo "Run: scripts/setup_dilightnet_baseline_env.sh $ENV_PREFIX"
  exit 1
fi

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  scripts/run_dilightnet_baseline_safe.sh <manifest.json> <output_root> [extra args...]

Example:
  scripts/run_dilightnet_baseline_safe.sh \
    logs/relighting_comparison/ra_manifest.json \
    logs/relighting_comparison/dilightnet_preds_ra \
    --staging-dir logs/relighting_comparison/dilightnet_staging_ra \
    --steps 20 --cfg 3.0 --prompt ""
EOF
  exit 1
fi

MANIFEST_PATH="$1"
OUTPUT_ROOT="$2"
shift 2

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_dilightnet_on_comparison_manifest.py" \
  --manifest "$MANIFEST_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --dilightnet-repo "$DILIGHTNET_REPO" \
  --python-bin "$PYTHON_BIN" \
  --run-official \
  "$@"
