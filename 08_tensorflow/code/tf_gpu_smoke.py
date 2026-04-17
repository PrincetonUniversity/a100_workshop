import importlib
from time import perf_counter


tf = importlib.import_module("tensorflow")


print(f"tensorflow={tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"gpus={gpus}")

if not gpus:
    raise SystemExit("No GPU detected by TensorFlow")

with tf.device("/GPU:0"):
    a = tf.random.normal((1024, 1024), dtype=tf.float32)
    b = tf.random.normal((1024, 1024), dtype=tf.float32)
    t0 = perf_counter()
    c = tf.matmul(a, b)
    _ = c.numpy()
    runtime = perf_counter() - t0

print(f"matmul_runtime={runtime:.6f}s")
