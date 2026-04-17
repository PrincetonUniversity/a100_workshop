#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/03_gpu_programming_review/code"

DRY_RUN=0
REQUIRE_NVC=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --require) REQUIRE_NVC=1 ;;
  esac
done

if [[ "${A100_WORKSHOP_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

compile_cmd=(nvc -acc -gpu=cc80 -Minfo=all -o vector_addition_openacc vector_addition.c)
run_cmd=(./vector_addition_openacc)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %s\n' "${TARGET_DIR}"
  printf '%s\n' "${compile_cmd[*]}"
  printf '%s\n' "${run_cmd[*]}"
  exit 0
fi

if ! command -v nvc >/dev/null 2>&1; then
  printf 'OpenACC wrapper skipped: nvc (NVIDIA HPC SDK) is not installed locally.\n' >&2
  if [[ "${REQUIRE_NVC}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

cd "${TARGET_DIR}"
"${compile_cmd[@]}"
"${run_cmd[@]}"
