# Takeaways

* The A100 GPU is a large improvement over the V100 for AI workloads. The performance of both FP64 and FP32 operations has increased as well as GPU memory bandwidth and capacity (40 or 80 GB).

* Always try to use the CUDA toolkit 11.x and cuDNN 8.x since they go with the A100. Using 10.x of the CUDA toolkit or 7.x of cuDNN may cause the GPU not to be used at. These libraries cannnot take advantage of the many features of the. For instance, a Conda install of TensorFlow will come with Cuda 10 and it will not use the GPU. See the PyTorch page for an example that uses 11.x.

* A100 very powerful so your application may not be able to utilize it sufficiently. In the case please move the work to another cluster. For real time monitoring on gpu utilization see this page.

* The A100 using compute capability 8.0. When compiling codes from source use `-arch=sm_80` or for CMake use `-DGPU_ARCH=sm_80` and `-DGMX_CUDA_TARGET_SM=80`. See the documentation of the application for the CMAKE option to specify. Note that the V100 used sm_70 and P100 using sm_60. If you are using a container from NGC that does not support sm_80 then consider using a different cluster. For instance, lammps with use the A100 while hoomd will not.

* Always start your jobs using 1 GPU per job. Conduct a [scaling analysis](https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis) to determine if additional GPUs are worth it.

* Mixed-precision algorithms that can take advantage of the Tensor Cores.

* Also, for AI frameworks much sure you use multiple data loading workers on the CPU to keep the GPU busy. See this video for the V100.
