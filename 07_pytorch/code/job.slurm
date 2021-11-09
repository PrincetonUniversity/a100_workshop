#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # v100 or a100
#SBATCH --reservation=a100       # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
singularity exec /scratch/network/jdh4/a100_wksp/containers/pytorch_21.10-py3.sif python myscript.py
