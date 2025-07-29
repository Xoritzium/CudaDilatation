from numba import cuda
import dilatation_kernel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import time
import os

class DilatationGPU:

    original_image = None
    grey_image_array = None

    images = []
    supported_images = (".jpg",".png", ".jpeg")
    output = "iteration,width,height,time\n"
    avg_solve_time = 0
    counter =0

    def __init__(self):
        self.load_images("./images")

        for i in range(11):
            for image in self.images:
                grey_image = ImageOps.grayscale(image)
                self.grey_image_array = np.asarray(grey_image)
                self.run_calculation(25,32,i)
                
        self.print_to_file()          

    def show_image(self,image,title):
        """
        visualise an given image with a given title as 'graph'
        close the window to continue the process
        """
        plt.figure()
        plt.imshow(image, cmap="grey")
        plt.title(title)
        plt.show()
    
    def run_calculation(self,mask, block_dimensions = 32, iteration = 0):
        """
        Run actual Cuda calculation with specified mask
        """
        g_img = cuda.to_device(self.grey_image_array) # transfer numpy array (x*x) to device
        g_result = cuda.device_array_like(self.grey_image_array)

        blockDimensions = (block_dimensions,block_dimensions)  # mmx threads in a block: 1024 or (32x32)
        gridDimensions = (self.grey_image_array.shape[0] // blockDimensions[0] +1, self.grey_image_array.shape[1] // blockDimensions[1] + 1)

        # actual calculation on decive (used: nvidia GForce 2070) 
        start = time.time()
        dilatation_kernel.dilatation_kernel[gridDimensions,blockDimensions](g_img,g_result,mask)
        end = time.time() - start

        final_result = g_result.copy_to_host() # reallocate the result back to the Host (cpu)
        self.output += f"{iteration},{self.grey_image_array.shape[0]},{self.grey_image_array.shape[1]}, {end} \n"
        cuda.current_context().deallocations.clear() # clear gpu memory
        

    def load_images(self,directory):
        #print(f"loading files from {directory}")
        for filename in os.listdir(directory):
            if filename.lower().endswith(self.supported_images):
                filepath = os.path.join(directory,filename)
                try:
                    self.images.append(Image.open((filepath)))
                except Exception as e:
                    print(f"error loading:  {filename}: {e}")                    
        #print(f"loaded files: {str(self.images)}")

    def print_to_file(self):
        with open("output.csv", "w", encoding="utf-8") as file:
            file.write(self.output)
    
