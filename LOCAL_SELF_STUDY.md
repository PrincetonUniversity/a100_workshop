# Local A100 Self-Study Guide

This repository was originally written for a Princeton workshop that used Slurm, cluster login nodes, and prebuilt module environments. If you already have a local machine with an NVIDIA A100, you can still use the repo as a structured self-study guide.

## What transfers directly

- the conceptual progression from `01_a100_overview` through `09_takeaways`
- the CUDA/OpenACC source files in `03_gpu_programming_review/code`
- the CuPy and PyTorch experiments in `06_cupy/code` and `07_pytorch/code`
- the A100-specific ideas around `sm_80`, TF32, Tensor Cores, and precision tradeoffs

## What is Princeton-specific

You can ignore these parts for local learning:

- `setup.md` cluster login and repository clone flow
- `sbatch job.slurm`
- `module load ...`
- `conda activate /scratch/network/...`
- hardcoded Singularity/container paths under Princeton storage

## Prerequisites

Before running anything locally, make sure:

```bash
nvidia-smi
```

You should see your A100 listed.

For the CUDA example you also need:

```bash
nvcc --version
```

For the Python examples you need a Python environment with GPU-enabled libraries that match your local driver/runtime.

## Suggested learning order

1. Read `README.md`
2. Read `01_a100_overview/README.md`
3. Read `03_gpu_programming_review/README.md` and inspect `03_gpu_programming_review/code/vector_addition.cu`
4. Run the CUDA example locally
5. Read `04_floating_point_formats/README.md`
6. Run the CuPy TF32 experiment locally
7. Run the PyTorch TF32 experiment locally
8. Read `08_tensorflow/README.md` and optionally run the local TensorFlow smoke test
9. Finish with `09_takeaways/README.md`

## Quick verification

Before doing real runs, verify the local wrappers:

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```

You can also inspect the commands each wrapper would run:

```bash
scripts/local/run_03_cuda.sh --dry-run
scripts/local/run_03_openacc.sh --dry-run
scripts/local/run_06_cupy.sh --dry-run
scripts/local/run_07_pytorch.sh --dry-run
scripts/local/run_08_tensorflow.sh --dry-run
```

## Local runs

### 03 CUDA

```bash
scripts/local/run_03_cuda.sh
```

### 03 OpenACC

Requires the NVIDIA HPC SDK (`nvc`). If it is not installed, the wrapper exits cleanly with a skip message.

```bash
scripts/local/run_03_openacc.sh
```

### 06 CuPy

The original workshop uses a very large matrix. For a quick local smoke run, start smaller:

```bash
CUPY_N=4096 CUPY_OUTER_REPS=1 CUPY_INNER_REPS=1 scripts/local/run_06_cupy.sh
```

To compare FP32 vs TF32 explicitly:

```bash
CUPY_TF32=0 CUPY_N=4096 CUPY_OUTER_REPS=1 CUPY_INNER_REPS=1 scripts/local/run_06_cupy.sh
CUPY_TF32=1 CUPY_N=4096 CUPY_OUTER_REPS=1 CUPY_INNER_REPS=1 scripts/local/run_06_cupy.sh
```

### 07 PyTorch

```bash
TORCH_N=4096 scripts/local/run_07_pytorch.sh
```

That single run already prints timings for both the default TF32-enabled path and the follow-up FP32 path with TF32 disabled.

### 08 TensorFlow

This repository did not originally include a local TensorFlow example, so the wrapper runs a small GPU smoke test added for workstation use:

```bash
scripts/local/run_08_tensorflow.sh
```

## Troubleshooting

### `nvidia-smi` works but Python cannot see the GPU

Your Python package may not match your installed CUDA/driver stack. Reinstall the framework build that matches your local environment.

### `nvcc` is missing

You likely have the driver installed but not the CUDA toolkit. Install a CUDA toolkit that supports A100 (`sm_80`).

### CuPy import fails

Install a CuPy build compatible with your CUDA runtime. The exact package name depends on your CUDA version.

### PyTorch runs on CPU only

Check `python -c "import torch; print(torch.cuda.is_available())"` and confirm you installed a CUDA-enabled PyTorch build.

### TensorFlow cannot find the GPU

TensorFlow GPU packaging is the most version-sensitive part of this repo. Treat the TensorFlow smoke test as optional unless you already have a known-good local TensorFlow GPU environment.
