# PyTorch

PyTorch will perform FP32 matrix multiplications using TF32 by default. Consider the code below:

```python
# https://pytorch.org/docs/stable/notes/cuda.html

import torch
from time import perf_counter

a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()  # 80.7277
print(mean)

a = a_full.float()
b = b_full.float()

# Do matmul at TF32 mode.
t0 = perf_counter()
ab_tf32 = a @ b  # takes 0.016s on GA100
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_tf32 - ab_full).abs().max()  # 0.1747
relative_error = error / mean  # 0.0022

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
t0 = perf_counter()
ab_fp32 = a @ b  # takes 0.11s on GA100
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_fp32 - ab_full).abs().max()  # 0.0031
relative_error = error / mean  # 0.000039
```

```
$ cd a100_workshop/07_pytorch/code
$ sbatch job.slurm
```


## AMP

## dlprof


# Useful Links

[data types](https://pytorch.org/docs/stable/tensor_attributes.html)  
[About TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html)  
[AMP](https://pytorch.org/docs/stable/amp.html)  
[AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html)  
[Blog Post on AMP Benchmarks](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)  
[NVIDIA Deepp Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)  
