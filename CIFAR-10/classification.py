from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < x.size:
        out[pos] = x[pos] + y[pos]

def add_with_cuda(x, y):
    out = np.empty_like(x)
    threads_per_block = 32
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](x, y, out)
    return out

# Example usage
x = np.random.rand(128).astype(np.float32)
y = np.random.rand(128).astype(np.float32)
result = add_with_cuda(x, y)
print(result)
