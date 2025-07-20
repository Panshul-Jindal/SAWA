import numpy as np
import cupy as cp
import time

# Define matrix size - you can adjust this value to see the impact
matrix_size = 1000

# Create large random matrices on the CPU
matrix_a_cpu = np.random.rand(matrix_size, matrix_size)
matrix_b_cpu = np.random.rand(matrix_size, matrix_size)

print(f"Performing matrix multiplication of size {matrix_size}x{matrix_size}")

# --- CPU Matrix Multiplication ---
start_time_cpu = time.time()
result_cpu = np.dot(matrix_a_cpu, matrix_b_cpu)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print(f"CPU Execution Time: {cpu_time:.6f} seconds")

# --- GPU Matrix Multiplication ---
# Transfer matrices to GPU memory
matrix_a_gpu = cp.array(matrix_a_cpu)
matrix_b_gpu = cp.array(matrix_b_cpu)

start_time_gpu = time.time()
result_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)
# Synchronize GPU to ensure all operations are complete before timing ends
cp.cuda.Device(0).synchronize()  #
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu
print(f"GPU Execution Time: {gpu_time:.6f} seconds")

# Compare results (optional, but good for verification)
# np.testing.assert_allclose(result_cpu, result_gpu.get(), rtol=1e-5, atol=1e-8)
# print("Results are approximately equal between CPU and GPU.")

# Determine which device was faster
if cpu_time < gpu_time:
    print(f"CPU was faster by {gpu_time / cpu_time:.2f} times.")
else:
    print(f"GPU was faster by {cpu_time / gpu_time:.2f} times.")


