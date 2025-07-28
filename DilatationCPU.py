import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import time

class DilatationCPU:

    original_image =None
    grey_image_array = None


    def __init__(self):
        # setup image        
        self.original_image = Image.open("wood.jpg")
        grey_image = ImageOps.grayscale(self.original_image)
        self.grey_image_array = np.asarray(grey_image)
        
    def dilatation (self, image, mask= 3):
        image_rows, image_cols = image.shape
        result = np.zeros([image_rows,image_cols])
        
        for x in range(image_rows):
            for y in range(image_cols):
                brightest = 0
                for i in range(-(mask//2),(mask//2)+1):
                    for j in range(-(mask//2),(mask//2)+1):
                        if (0 <= x + i < image_rows) and (0 <= y + j < image_cols):
                            current = image[x + i, y + j]
                        else:
                            current = 0
                        if (current > brightest):
                            brightest = current
                result[x,y] =brightest

        return result

    def show_image(self,image,title):
        """
        visualise an given image with a given title as 'graph'
        For the process to continue, close the opened window!
        """
        plt.figure()
        plt.imshow(image, cmap="grey")
        plt.title(title)
        plt.show()
    
    def run_calculation(self, mask):
        """
        Runs actual calculation with the given mask
        also stops time, but since its only on cpu, it will take minutes
        """
        start = time.time()
        dilatation_image = self.dilatation(self.grey_image_array, mask)
        end = time.time() -start
        self.show_image(self.original_image, "original image")
        self.show_image(dilatation_image, "dilataion")
        print(f"solving with CPU took: {end/60} minutes")


        






    
    