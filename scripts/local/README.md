# Local wrapper scripts

These scripts provide a local-machine path through the workshop without Slurm or Princeton module commands.

## Available wrappers

- `scripts/local/run_03_cuda.sh` — compile and run the CUDA vector addition example with `nvcc`
- `scripts/local/run_03_openacc.sh` — compile and run the OpenACC example with `nvc`
- `scripts/local/run_06_cupy.sh` — run the CuPy TF32 experiment locally
- `scripts/local/run_07_pytorch.sh` — run the PyTorch TF32 experiment locally
- `scripts/local/run_08_tensorflow.sh` — run a minimal TensorFlow GPU smoke test locally

## Shared options

- `--dry-run` prints the commands without executing them
- `A100_WORKSHOP_DRY_RUN=1` enables dry-run mode through the environment

## Shared environment knobs

- `CUPY_N`, `CUPY_OUTER_REPS`, `CUPY_INNER_REPS`
- `TORCH_N`

The wrappers assume your local Python environment already has the required packages installed.
