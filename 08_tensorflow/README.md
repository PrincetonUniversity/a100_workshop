# TensorFlow

## Installing

Be careful. A Conda install from the main channel pulls in `cudatoolkit-10.x` and `cudnn-7.x`:

```
$ ssh <YourNetID>@della-gpu.princeton.edu
$ module load anaconda3/2020.11
$ conda create --name tf2-gpu tensorflow-gpu  # DO NOT DO THIS
...
cudatoolkit        pkgs/main/linux-64::cudatoolkit-10.1.243-h6bb024c_0
cudnn              pkgs/main/linux-64::cudnn-7.6.5-cuda10.1_0
...
```

See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow) page for the installation directions for the A100 clusters.

## Tensor Cores

TensorFlow using TF32 in place a single precision for matrix-matrix multiplications and convolutions by default. [Mixed precision](https://www.tensorflow.org/guide/mixed_precision).

```
#!/bin/bash
#SBATCH --job-name=tf2-ngc       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100

module purge
singularity exec --nv ../../a100_wksp/containers/tensorflow_21.10-tf2-py3.sif python3 mnist_classify.py
```

You should see the following in the output file:

```
TensorFloat-32 will be used for the matrix multiplication
```

This is an indication of the FP32 MM being done in TF32.
