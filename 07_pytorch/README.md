# PyTorch

PyTorch will perform FP32 matrix multiplications using TF32 by default. Consider the code below:

```python
# https://pytorch.org/docs/stable/notes/cuda.html

import torch
from time import perf_counter

a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()
print(mean)

a = a_full.float()
b = b_full.float()

# Do matmul at TF32 mode.
t0 = perf_counter()
ab_tf32 = a @ b  # takes 0.016s on GA100
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_tf32 - ab_full).abs().max()
relative_error = error / mean

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
t0 = perf_counter()
ab_fp32 = a @ b  # takes 0.11s on GA100
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_fp32 - ab_full).abs().max()
relative_error = error / mean
```

```
$ cd a100_workshop/07_pytorch/code
$ sbatch job.slurm
```

## Automatic Mixed Precision (AMP)

Mixed-precision training in PyTorch is done through [AMP](https://pytorch.org/docs/stable/amp.html).

## dlprof

The deep learning profiler from NVIDIA is called `dlprof`. See a sample script [here](https://github.com/PrincetonUniversity/gpu_programming_intro/blob/master/04_gpu_tools/README.md#dlprof).

# Useful Links

[PyTorch Data Types](https://pytorch.org/docs/stable/tensor_attributes.html)  
[About TensorFloat32 in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html)  
[AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html)  
[Blog Post on AMP Benchmarks](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)  
[NVIDIA Deepp Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)  
