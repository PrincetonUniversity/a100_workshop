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
ab_tf32 = a @ b
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_tf32 - ab_full).abs().max()
relative_error = error / mean

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
t0 = perf_counter()
ab_fp32 = a @ b
torch.cuda.synchronize()
print(perf_counter() - t0)
error = (ab_fp32 - ab_full).abs().max()
relative_error = error / mean
