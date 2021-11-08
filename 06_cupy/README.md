# CuPy

NumPy is a Python library for numerical computing on CPUs. CuPy is a drop-in replacement for NumPy on GPUs:

### NumPy

```python
import numpy as np
X = np.random.randn(100, 100)
u, s, v = np.linalg.svd(X)
```

### CuPy

```python
import cupy as cp
X = cp.random.randn(100, 100)
u, s, v = cp.linalg.svd(X)
``

CuPy is similar to NumPy except it is designed for GPUs instead of CPUs.


```bash
$ module load anaconda3/2020.11
$ conda create --name cupy-env cupy --channel conda-forge
```

Make sure you get CUDA toolkit 11.x and CuPy > 8.x:

```
  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-1_gnu
  ca-certificates    conda-forge/linux-64::ca-certificates-2021.10.8-ha878542_0
  cudatoolkit        conda-forge/linux-64::cudatoolkit-11.5.0-h36ae40a_9
  cupy               conda-forge/linux-64::cupy-9.5.0-py39h499daff_0
  fastrlock          conda-forge/linux-64::fastrlock-0.8-py39he80948d_0
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.36.1-hea4e1c9_2
  libblas            conda-forge/linux-64::libblas-3.9.0-12_linux64_openblas
  libcblas           conda-forge/linux-64::libcblas-3.9.0-12_linux64_openblas
  libffi             conda-forge/linux-64::libffi-3.4.2-h9c3ff4c_4
  libgcc-ng          conda-forge/linux-64::libgcc-ng-11.2.0-h1d223b6_11
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-11.2.0-h69a702a_11
  libgfortran5       conda-forge/linux-64::libgfortran5-11.2.0-h5c6108e_11
  libgomp            conda-forge/linux-64::libgomp-11.2.0-h1d223b6_11
  liblapack          conda-forge/linux-64::liblapack-3.9.0-12_linux64_openblas
  libopenblas        conda-forge/linux-64::libopenblas-0.3.18-pthreads_h8fe5266_0
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-11.2.0-he4da1e4_11
  libzlib            conda-forge/linux-64::libzlib-1.2.11-h36c2ea0_1013
  ncurses            conda-forge/linux-64::ncurses-6.2-h58526e2_4
  numpy              conda-forge/linux-64::numpy-1.21.3-py39hdbf815f_0
  openssl            conda-forge/linux-64::openssl-3.0.0-h7f98852_1
  pip                conda-forge/noarch::pip-21.3.1-pyhd8ed1ab_0
  python             conda-forge/linux-64::python-3.9.7-hf930737_3_cpython
  python_abi         conda-forge/linux-64::python_abi-3.9-2_cp39
  readline           conda-forge/linux-64::readline-8.1-h46c0cb4_0
  setuptools         conda-forge/linux-64::setuptools-58.2.0-py39hf3d152e_0
  sqlite             conda-forge/linux-64::sqlite-3.36.0-h9cd32fc_2
  tk                 conda-forge/linux-64::tk-8.6.11-h27826a3_1
  tzdata             conda-forge/noarch::tzdata-2021e-he74cb21_0
  wheel              conda-forge/noarch::wheel-0.37.0-pyhd8ed1ab_1
  xz                 conda-forge/linux-64::xz-5.2.5-h516909a_1
  zlib               conda-forge/linux-64::zlib-1.2.11-h36c2ea0_1013
```

CuPy can be made use TensorFloat32 for FP32 matrix-matrix multiplies:

```python
import cupy as cp
from time import perf_counter

N = 15000
X = cp.random.randn(N, N, dtype=cp.float32)
Y = cp.random.randn(N, N, dtype=cp.float32)
Z = cp.matmul(X, Y)  # compile the kernel

times = []
for _ in range(3):
  t0 = perf_counter()
  for _ in range(10):
    Z = cp.matmul(X, Y)
    cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - t0)
print(min(times))
```

```bash
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # v100 or a100

module purge
module load anaconda3/2020.11
conda activate /scratch/network/jdh4/CONDA/envs/cupy-env  # make in a100-wksp directory

# python myscript.py
# CUPY_TF32=0 python myscript.py
CUPY_TF32=1 python myscript.py
```

Below are the results:

| GPU                  | CUPY_TF32  | execution time (s)  |  speed-up factor   |
|:--------------------:|:------------------:|:-----------:|:------------------:|
|  V100                | N/A                |    5.3      |      1.0           |
|  A100                | 0                  |    4.0      |      1.3           |
|  A100                | 1                  |    0.8      |      6.5           |


Rows 1 and 2 indicate that the A100 gives a speed-up of 1.3x over the V100. When the computation is performed in TensorFloat32 the speed-up is 6.5x. Is there a loss of precision?

```
$ salloc -N 1 -n 1 -t 5 --gres=gpu:1 --constraint=a100
$ module load anaconda3/2020.11
$ conda activate /scratch/network/jdh4/CONDA/envs/cupy-env
$ env CUPY_TF32=0 CUPY_SEED=42 python
>>> import cupy as cp
>>> X = cp.random.randn(3, 3, dtype=cp.float32)
>>> cp.matmul(X, X)
array([[ 3.3182788 ,  1.1980604 ,  1.2314391 ],
       [-0.44799405,  4.8557844 , -1.0284786 ],
       [ 1.932775  , -3.3323417 ,  3.766266  ]], dtype=float32)
>>> exit()
```

```
$ env CUPY_TF32=1 CUPY_SEED=42 python
>>> import cupy as cp
>>> X = cp.random.randn(3, 3, dtype=cp.float32)
>>> cp.matmul(X, X)
array([[ 3.316953  ,  1.1977959 ,  1.2309666 ],
       [-0.44838548,  4.8550496 , -1.0283203 ],
       [ 1.933102  , -3.3319902 ,  3.7654052 ]], dtype=float32)
```

We see that indeed the numbers are different so there is some loss of precision.

```
/scratch/network/jdh4/CONDA/envs/cupy-env/lib/python3.9/site-packages/cupy/_core/_gufuncs.py:225: UserWarning: COMPUTE_TYPE_BF16 and COMPUTE_TYPE_TF32 are only available on GPUs with compute capability 8.0 or higher. COMPUTE_TYPE_DEFAULT will be used instead.
```

More can be done with [CUB and cuTensor](https://tech.preferred.jp/en/blog/cupy-v8/). CuPy can do half-precision FFTs.

Installation

This can be done via Conda or singularity:

```
$ singularity pull docker://cupy/cupy:v9.5.0
```
