# Floating Point Formats and Tensor Cores

By reducing the number of bits to represent a floating-point number, less data needs to the transferred and stored. This leads to a performance boost. NVIDIA has introduced a number of different floating point formats that can be used in the Tensor Cores in the A100 GPU.

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
#SBATCH --constraint=a100        # v100 or a100

./bf16TensorCoreGemm
```

The job above cannot run on the V100 node since the V100 GPU does not support bfloat16.
