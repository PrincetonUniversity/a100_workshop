# Takeaways

* The A100 GPU is a large improvement over the V100 for AI workloads. The performance of both FP64 and FP32 operations has increased as well as GPU memory bandwidth and capacity (40 or 80 GB).

* Always try to use the CUDA Toolkit 11.x and cuDNN 8.x since they are needed to take full advantage of the A100. Using 10.x of the CUDA toolkit or 7.x of cuDNN may cause the GPU not to be used at all. For instance, a naive Conda install of TensorFlow will come with cudatoolkit-10.x and it will not use the GPU. See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow) page for the correct directions.

* A100 very powerful so your application may not be able to utilize it sufficiently. In that case, please move the work to another cluster. For real-time monitoring on GPU utilization see our [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page.

* The A100 uses compute capability 8.0. When compiling codes from source use `-arch=sm_80` or for CMake use `-DGPU_ARCH=sm_80` and `-DGMX_CUDA_TARGET_SM=80`. See the documentation of the application for the CMAKE option to specify. Note that the V100 uses `sm_70` and P100 uses `sm_60`. If you are using a container from NGC that does not support sm_80 then consider using a different cluster.

* Always start your jobs using 1 GPU per job. Conduct a [scaling analysis](https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis) to determine if additional GPUs are worth it.
