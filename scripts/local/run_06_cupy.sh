#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/06_cupy/code"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]] || [[ "${A100_WORKSHOP_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

run_cmd=(python myscript.py)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %s\n' "${TARGET_DIR}"
  printf '%s\n' "python -c \"import cupy as cp; count = cp.cuda.runtime.getDeviceCount(); print(count)\""
  printf '%s\n' "${run_cmd[*]}"
  exit 0
fi

command -v python >/dev/null 2>&1 || {
  printf 'Missing required executable: python\n' >&2
  exit 1
}

cd "${TARGET_DIR}"
python -c "import cupy as cp; count = cp.cuda.runtime.getDeviceCount(); print(count); (_ for _ in ()).throw(SystemExit('No CUDA device visible to CuPy')) if count < 1 else None"
"${run_cmd[@]}"
