# Floating Point Formats and Tensor Cores

A100 has tensor cores for fp64 as well.

By reducing the number of bits to represent a floating-point number, less data needs to the transferred and stored. This leads to a performance advantage.

The V100 GPUs have 640 Tensor Cores (8 per streaming multiprocessor) where half-precision (16 bits FP16) Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each Tensor Core can multiply two 4 x 4 matrices together in half-precision and add the result to a third matrix which is in full precision. This is useful for training and inference on deep neural networks and many other computations that are rooted in linear algebra.

There are several use cases where the Tensor Cores can be utilized on the V100 GPUs of Traverse. In general it is algorithms that use Level 3 BLAS routines. In almost all cases the user needs to explicitly take action to use the Tensor Cores.

The NVIDIA Apex library allows for automatic mixed-precision (AMP) training and distributed training of neural networks. It is included with an installation of PyTorch from WML-CE. To see the performance benefit of the Tensor Cores, download the dcgan example and run it with and without using the Tensor Cores. Using 16 hardware threads one finds a speed-up of about 10%. Note that to use the fp16 kernels the dimension of each matrix must be a multiple of 8. Read about the constraints here.

Another example using Fortran is here. There are algorithms in the MAGMA library (discussed below) that can utilize the Tensor Cores of V100 GPUs. Mixed precision Krylov and Multigrid solvers have also been developed, as discussed in this presentation.

NVIDIA has introduced a larger number and different types of Tensor Cores in the A100 GPU. Additionally, in many cases the Tensor Cores are automatically used and many of the constraints have been relaxed. There are no Tensor Cores on the P100 GPUs on TigerGPU.

AMD calls them Matrix Cores.

![precision](https://blogs.nvidia.com/wp-content/uploads/2020/05/tf32-Mantissa-chart-hi-res-FINAL.png.webp)

# HPC

NVIDIA has made the Tensor Cores routines available through the CUDA API. This means that developers can take advantage of these specialized hardware units for workloads unrelated to deep learning.

Follow these directions to see the examples:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/gpfs/$USER
$ /usr/local/cuda-11.4/bin/cuda-install-samples-11.4.sh .
$ cd NVIDIA_CUDA-11.4_Samples/0_Simple
$ ls -d *TensorCore*
bf16TensorCoreGemm
cudaTensorCoreGemm
dmmaTensorCoreGemm
immaTensorCoreGemm
tf32TensorCoreGemm
$ module load cudatoolkit/11.4
$ cd bf16TensorCoreGemm
$ make
$ sbatch job.slurm  # create this file from below
```

Below is a sample Slurm script:

```
#!/bin/bash
#SBATCH --job-name=tensor-cores  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (default is 4 GB per CPU-core)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100

./bf16TensorCoreGemm
```

The job above cannot run on the V100 node since the V100 GPU does not support bfloat16.
