#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/run_dilightnet_baseline_safe.sh"
OUT_ROOT="${1:-${REPO_ROOT}/logs/relighting_comparison/dilightnet_preds}"
SHIFTED=0
if [[ $# -ge 1 ]]; then
  SHIFTED=1
fi
if [[ "$SHIFTED" -eq 1 ]]; then
  shift
fi

bash "$RUNNER" \
  "${REPO_ROOT}/logs/relighting_comparison/ra_manifest.json" \
  "${OUT_ROOT}/ra" \
  --staging-dir "${REPO_ROOT}/logs/relighting_comparison/dilightnet_staging_ra" \
  "$@"

bash "$RUNNER" \
  "${REPO_ROOT}/logs/relighting_comparison/uu_manifest.json" \
  "${OUT_ROOT}/uu" \
  --staging-dir "${REPO_ROOT}/logs/relighting_comparison/dilightnet_staging_uu" \
  "$@"

bash "$RUNNER" \
  "${REPO_ROOT}/logs/relighting_comparison/us_manifest.json" \
  "${OUT_ROOT}/us" \
  --staging-dir "${REPO_ROOT}/logs/relighting_comparison/dilightnet_staging_us" \
  "$@"
