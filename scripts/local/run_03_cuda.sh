#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/03_gpu_programming_review/code"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]] || [[ "${A100_WORKSHOP_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

compile_cmd=(nvcc -O3 -arch=sm_80 -o vector_addition vector_addition.cu)
run_cmd=(./vector_addition)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %s\n' "${TARGET_DIR}"
  printf '%s\n' "${compile_cmd[*]}"
  printf '%s\n' "${run_cmd[*]}"
  exit 0
fi

command -v nvcc >/dev/null 2>&1 || {
  printf 'Missing required executable: nvcc\n' >&2
  exit 1
}

cd "${TARGET_DIR}"
"${compile_cmd[@]}"
"${run_cmd[@]}"
