# CuPy

[NumPy](https://numpy.org) is a Python library for numerical computing on CPUs. [CuPy](https://cupy.dev) is a drop-in replacement for NumPy on GPUs:

NumPy code runs on CPUs:

```python
import numpy as np
X = np.random.randn(100, 100)
u, s, v = np.linalg.svd(X)
```

CuPy code runs on GPUs:

```python
import cupy as cp
X = cp.random.randn(100, 100)
u, s, v = cp.linalg.svd(X)
```

## CuPy uses Tensor Cores

CuPy can take advantage of the A100 since the release of version 8 of CuPy. It can be made use TensorFloat32 for FP32 matrix-matrix multiplies. However, by default this is turned off. Consider the following Python code below:

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

Follow the directions below to run the code above on the V100 and A100. We will consider two cases on the A100: one with TF32 and one without.

#### Case 1

Run the code on the V100 GPU.

```
$ cd a100_workshop/05_cupy
```

Then make sure that the `v100` will be used:

```
#SBATCH --constraint=v100
```

And make the last line of `job.slurm`:

```
python myscript.py
```

Submit the job and record the run time:

```
$ sbatch job.slurm
```

#### Case 2

Run the code on the A100 GPU with TF32 turned off. Modify `job.slurm` as follows:

```
#SBATCH --constraint=a100
```

And the last line should be:

```
CUPY_TF32=0 python myscript.py
```


#### Case 3

Run the code on the A100 GPU with TF32 turned off. Modify `job.slurm` as follows:

```
#SBATCH --constraint=a100
```

And the last line should be:

```
CUPY_TF32=1 python myscript.py
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
#SBATCH --reservation=a100       # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2020.11
conda activate /scratch/network/jdh4/CONDA/envs/cupy-env

# python myscript.py               # case 1
# CUPY_TF32=0 python myscript.py   # case 2
# CUPY_TF32=1 python myscript.py   # case 3
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

## Installation

You do not need to install anything for the live workshop. Later on you can use these commands to install CuPy:

#### Conda

Make sure you get CUDA Toolkit 11.x to take full advantage of the A100 GPUs:

```bash
$ module load anaconda3/2020.11
$ conda create --name cupy-env cupy --channel conda-forge
```

When installing via Anaconda or using a container, make sure you get the software built against CUDA Toolkit 11.x. For instance, with Conda:

```
  ...
  cudatoolkit        conda-forge/linux-64::cudatoolkit-11.5.0-h36ae40a_9
  cupy               conda-forge/linux-64::cupy-9.5.0-py39h499daff_0
  ...
```

For more on Conda environments see [Python on the HPC Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/python).

#### Singularity

Find the tag for the desired version on the [CuPy DockerHub repo](https://hub.docker.com/r/cupy/cupy) then pull the container:

```
$ singularity pull docker://cupy/cupy:v9.5.0
```

Learn more about [Singularity](https://researchcomputing.princeton.edu/support/knowledge-base/singularity).
