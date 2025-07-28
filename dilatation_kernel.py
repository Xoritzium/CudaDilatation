from numba import cuda

@cuda.jit
def dilatation_kernel(image, result_Img, mask = 3):
    """
    GPU kernel to apply a basic dilatation mask on the given grayscaled image
    """
    x,y =  cuda.grid(2) # get the apsolute position of the thread, based on cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x, analog for y
    image_rows, image_cols = image.shape 
    if( x >= image_rows or y >= image_cols):
        raise Exception("grid calculation failed")
    else:
        brightest = 0
        for i in range(-(mask//2),(mask//2)+1): # could be optimized, but for sake of understanding the dilatation its kept this way
            for j in range(-(mask//2),(mask//2)+1):
                if  (x+i >=0 and x+i <= image_rows) and (y+j >= 0 and y+j <= image_cols):
                    current = image[x+i,y+j]
                else: # ignore border cases
                    current = 0
                if(current > brightest):
                    brightest = current        
        result_Img[x,y] = brightest # since the image and the result_img matrix have the same size, the corresponding pixel representation matches



    x,y = cuda.grid(2)
    result_Img[x,y] = image[x,y] # example: copy image