# CUDA Multi-Process Service (MPS)

Certain MPI codes that use GPUs may benefit from CUDA MPS (see [ORNL docs](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#volta-multi-process-service)), which enables multiple processes to concurrently share the resources on a single GPU. To use MPS simply add this directive to your Slurm script (`della-gpu` and `traverse`):

```
#SBATCH --gpu-mps
```

In most cases users will see no speed-up. Codes where the individual MPI processes underutilize the GPU should see a performance gain. This NVIDIA [blog post](https://developer.nvidia.com/blog/maximizing-gromacs-throughput-with-multiple-simulations-per-gpu-using-mps-and-mig/) details the use of MPS and MIG for GROMACS.


# GPUDirect

Using [GPUDirect](https://developer.nvidia.com/gpudirect), multiple GPUs, network adapters, solid-state drives and NVMe drives can directly read and write CUDA host and device memory, eliminating unnecessary memory copies, dramatically lowering CPU overhead, and reducing latency, resulting in significant performance improvements in data transfer times for applications running on NVIDIA GPUs.

![gpu-direct](https://developer.nvidia.com/sites/default/files/akamai/GPUDirect/cuda-gpu-direct-blog-refresh_diagram_1.png)

On `della-gpu` we have [GDRCopy](https://github.com/NVIDIA/gdrcopy) installed (see `/lib64/libgdrapi.so`). GPUDirect is also available on Traverse.


# CUDA-Aware MPI

In addition to traditional builds of the MPI library, there are CUDA-aware MPI builds which allow for data on a GPU to be sent to another GPU without going through a CPU. Regular MPI implementations pass pointers to host memory, staging GPU buffers through host memory using cudaMemcopy. With CUDA-aware MPI, the MPI library can send and receive GPU buffers directly, without having to first stage them in host memory. To see the CUDA-aware MPI modules:

```
$ module avail openmpi/cuda
```

See an [example](https://github.com/PrincetonUniversity/hpc_beginning_workshop/tree/2021fall/RC_example_jobs/cuda_mpi) of a simple code that uses CUDA-aware MPI.
