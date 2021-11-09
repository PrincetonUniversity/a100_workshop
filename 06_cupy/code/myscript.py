import cupy as cp
from time import perf_counter

N = 15000
X = cp.random.randn(N, N, dtype=cp.float32)
Y = cp.random.randn(N, N, dtype=cp.float32)
Z = cp.matmul(X, Y)  # compile the kernel

times = []
for _ in range(3):
  t0 = perf_counter()
  for _ in range(10):
    Z = cp.matmul(X, Y)
    cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - t0)
print(min(times))
