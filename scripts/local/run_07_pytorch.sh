#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/07_pytorch/code"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]] || [[ "${A100_WORKSHOP_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

run_cmd=(python myscript.py)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %s\n' "${TARGET_DIR}"
  printf '%s\n' "python -c \"import torch; print(torch.cuda.is_available()); print(torch.version.cuda)\""
  printf '%s\n' "${run_cmd[*]}"
  exit 0
fi

command -v python >/dev/null 2>&1 || {
  printf 'Missing required executable: python\n' >&2
  exit 1
}

cd "${TARGET_DIR}"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); (_ for _ in ()).throw(SystemExit('PyTorch is installed but CUDA is not available. Install a CUDA-enabled build.')) if not torch.cuda.is_available() else None"
"${run_cmd[@]}"
