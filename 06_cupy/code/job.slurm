#!/bin/bash
#SBATCH --job-name=cupy          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G               # memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=v100        # v100 or a100

module purge
module load anaconda3/2022.5
conda activate /scratch/network/jdh4/.gpu_workshop/envs/cupy-env

echo "GPU is " $(nvidia-smi -a | grep "Product Name" | awk '{print $(NF)}')

python myscript.py               # case 1
# CUPY_TF32=0 python myscript.py   # case 2
# CUPY_TF32=1 python myscript.py   # case 3
