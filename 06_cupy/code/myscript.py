import os
import cupy as cp
from time import perf_counter

n = int(os.environ.get("CUPY_N", "15000"))
outer_reps = int(os.environ.get("CUPY_OUTER_REPS", "3"))
inner_reps = int(os.environ.get("CUPY_INNER_REPS", "10"))

x = cp.random.randn(n, n, dtype=cp.float32)
y = cp.random.randn(n, n, dtype=cp.float32)
z = cp.matmul(x, y)  # compile the kernel

times = []
for _ in range(outer_reps):
    t0 = perf_counter()
    for _ in range(inner_reps):
        z = cp.matmul(x, y)
        cp.cuda.Device(0).synchronize()
    times.append(perf_counter() - t0)
print("execution time =", min(times))

if "CUPY_TF32" in os.environ:
    print("Using TF32") if int(os.environ["CUPY_TF32"]) else print("Using FP32")
else:
    print("Using FP32")
