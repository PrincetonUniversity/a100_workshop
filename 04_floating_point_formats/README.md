# Tensor Cores

A100 has tensor cores for fp64 as well.

By reducing the number of bits to represent a floating-point number, less data needs to the transferred and stored. This leads to a performance advantage.

The V100 GPUs have 640 Tensor Cores (8 per streaming multiprocessor) where half-precision (16 bits FP16) Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each Tensor Core can multiply two 4 x 4 matrices together in half-precision and add the result to a third matrix which is in full precision. This is useful for training and inference on deep neural networks and many other computations that are rooted in linear algebra.

There are several use cases where the Tensor Cores can be utilized on the V100 GPUs of Traverse. In general it is algorithms that use Level 3 BLAS routines. In almost all cases the user needs to explicitly take action to use the Tensor Cores.

The NVIDIA Apex library allows for automatic mixed-precision (AMP) training and distributed training of neural networks. It is included with an installation of PyTorch from WML-CE. To see the performance benefit of the Tensor Cores, download the dcgan example and run it with and without using the Tensor Cores. Using 16 hardware threads one finds a speed-up of about 10%. Note that to use the fp16 kernels the dimension of each matrix must be a multiple of 8. Read about the constraints here.

Another example using Fortran is here. There are algorithms in the MAGMA library (discussed below) that can utilize the Tensor Cores of V100 GPUs. Mixed precision Krylov and Multigrid solvers have also been developed, as discussed in this presentation.

NVIDIA has introduced a larger number and different types of Tensor Cores in the A100 GPU. Additionally, in many cases the Tensor Cores are automatically used and many of the constraints have been relaxed. There are no Tensor Cores on the P100 GPUs on TigerGPU.

AMD calls them Matrix Cores.

![precision](https://blogs.nvidia.com/wp-content/uploads/2020/05/tf32-Mantissa-chart-hi-res-FINAL.png.webp)

# Numerical Precision
