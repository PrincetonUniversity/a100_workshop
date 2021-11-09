# GPU Programming Review

Here we compile and run two GPU codes using the two most popular programming models, namely, CUDA and OpenACC. Previous experience is not necessary. We will consider the case of element-wise vector addition.

## CUDA (Hands-on Exercise)

The CUDA code is available within this repo:

```bash
$ cd a100_workshop/03_programming_review/code
```

Identify the lines were (1) the data is being sent to the GPU, (2) the GPU kernel is called and (3) the lines where the data is copied back from the GPU to the CPU:

```
$ vim vector_addition.cu  # or emacs, micro, nano, cat
```

Run the following commands to compile and run the code on the A100 GPU:

```bash
$ module load cudatoolkit/11.4
$ nvcc -O3 -arch=sm_80 -o vector_addition vector_addition.cu  # sm_70 for V100, sm_80 for A100
$ sbatch job.slurm
$ cat slurm-*.out
```

Note that you can build an executable for both the V100 and A100 with `-arch=sm_70,sm_80`.

Don't worry about measuring the performance difference between the CPU and the GPU. The purpose of this exercise is simply to revisit CUDA.

## Aside on GPU Utilization

Read about [Measuring GPU Utilization in Real Time](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpu-utilization). An example of a code that shows 100% GPU utilization but is only using a single thread is `a100_workshop/programming_review/code/spurious_high_utilization.cu`.

To really know how effectively a GPU is being utilized one must measure the GPU occupancy using a profiler like [Nsight Compute](https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/04_gpu_tools#nsight-compute-ncu-for-gpu-kernel-profiling).

## OpenACC (Hands-on Exercise)

[OpenACC](https://www.openacc.org) is a high-level programming model where compiler directives are used to port CPU code to the GPU.

Use a text editor to modify `vector_addition.c` as follows (add lines in bold):

<pre>
...
#include <math.h>
#include "timer.h"
<b>#include "openacc.h"</b>

void vecAdd(double *a, double *b, double *c, int n)
{
    int i;
    <b>#pragma acc parallel loop</b>
    for(i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
...
</pre>

Compile and run the code:

```bash
$ module load nvhpc/21.5
$ module load cudatoolkit/11.4
$ nvc -acc -gpu=cc70,cc80 -Minfo=all -o vector_addition vector_addition.c
$ sbatch job.slurm
```

## Useful Links

[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[CUDA C++ Programming Guide by NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[www.openacc.org](https://www.openacc.org)  
