// The purpose of this code is to illustrate that 100% GPU utilization as
// measured using nvidia-smi does not necessarily mean that the GPU is being
// efficiently used. In this case only a single thread is used.

#include <stdio.h>

void CPUFunction() {
  printf("\"Hello world\" from the CPU.\n");
}

__global__ void GPUFunction() {
  printf("\"Hello\" from the one and only GPU thread (index %d).\n", threadIdx.x);
  while (true) {}; // infinite loop
}

int main() {
  // function to run on the cpu
  CPUFunction();

  // function to run on the gpu
  GPUFunction<<<1, 1>>>();
  
  // kernel execution is asynchronous so sync on its completion
  cudaDeviceSynchronize();
}
