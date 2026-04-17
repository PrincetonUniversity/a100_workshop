#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/08_tensorflow/code"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]] || [[ "${A100_WORKSHOP_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

run_cmd=(python tf_gpu_smoke.py)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %s\n' "${TARGET_DIR}"
  printf '%s\n' "python -c \"import tensorflow as tf; print(tf.__version__)\""
  printf '%s\n' "${run_cmd[*]}"
  exit 0
fi

command -v python >/dev/null 2>&1 || {
  printf 'Missing required executable: python\n' >&2
  exit 1
}

cd "${TARGET_DIR}"
python -c "import tensorflow as tf; print(tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(gpus); (_ for _ in ()).throw(SystemExit('TensorFlow is installed but no GPU is visible.')) if not gpus else None"
"${run_cmd[@]}"
