from numba import cuda
import numpy as np
from PIL import Image, ImageOps
import time
from math import exp
import matplotlib.pyplot as plt


@cuda.jit # compile into a gpu kernel (jit = just in time)
def gaussion_gpu(sigma,kernel_size,kernel):
    """
    creates the gaussian blur in "kernerl"
    """
    m = kernel_size //2
    n = kernel_size // 2

    x = cuda.threadIdx.x # current position of a thread IN A BLOCK!
    y = cuda.threadIdx.y
        # exp is most likely the gaussion calculation
    kernel[x,y] = exp(-((x-m) ** 2 +(y-n)**2) / (2*sigma **2))  # write into device (gpu) memory


@cuda.jit
def convolve(result,mask,image):
    i,j = cuda.grid(2) # compute  compute grid dimensions
    image_rows, image_cols = image.shape
    if( i >= image_rows) or (j >= image_cols):
        return
    
    delta_rows = mask.shape[0] // 2
    delta_cols = mask.shape[1] // 2

    s = 0
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            if(i_k >=0) and (i_k< image_rows) and (j_l >= 0)and (j_l <= image_cols):
                s += mask[k,l] * image[i_k,j_l]
    result[i,j] = s



sigma = 2
kernel_size = 30
kernel = np.zeros((kernel_size,kernel_size), np.float32)
d_kernel = cuda.to_device(kernel) # move data to device


plainImage = Image.open('wood.jpg')
image = np.asarray(ImageOps.grayscale(plainImage))
print(image)
d_image = cuda.to_device(image)
d_result = cuda.device_array_like(image)

gaussion_gpu[(1,), (kernel_size,kernel_size)](sigma, kernel_size, d_kernel)

blockdim = (32,32)
griddim =(image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] +1)

start = time.process_time()
convolve[griddim,blockdim] (d_result,d_kernel,d_image)
print(time.process_time()-start)
result = d_result.copy_to_host()

plt.figure()
plt.imshow(image, cmap="gray")
plt.title = ("before convolution:")

plt.figure()
plt.imshow(result, cmap="gray")
plt.title = ("after convolution:")
plt.show()
 



