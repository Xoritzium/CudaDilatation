from numba import cuda
import numpy as np

@cuda.jit
def cuda_kernel(array,result):
    # print different things
    tx = cuda.threadIdx.x # thread id in 1D block
    ty = cuda.blockIdx.y # block id in 1D grid
    bw = cuda.blockDim.x # block width = threadcound within this block
    
    # array position in relation to the grid structure:
    pos = tx + ty * bw
    if pos < len(arr): # always check boundaries if len(arr) % block AND grid size = 0
        result[pos] =array[pos]


arr = np.array([1,2,3,4])
res = np.zeros(4, dtype =int)
# set kernel dimensions
threadsPerBlock = 32
blocksperGrid = (len(arr)+ (threadsPerBlock - 1)) // threadsPerBlock # will be resolved into one block
print("Arraysize: " + str(len(arr)) + ", threads per Block: " + str(threadsPerBlock) +", blocks per Grid: " + str(blocksperGrid))
print("array: " + str(arr))
print("result: " +  str(res))
# send to gpu memory
g_res = cuda.to_device(res)
g_arr = cuda.to_device(arr)
# invoke kernel
cuda_kernel[blocksperGrid,threadsPerBlock] (g_arr, g_res)

# copy values back to print the result.
result = g_res.copy_to_host()
print("final result: " + str(result))