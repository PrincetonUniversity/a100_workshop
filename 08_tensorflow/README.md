# TensorFlow

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
