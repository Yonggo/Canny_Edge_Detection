import Gradient
from numba import jit, cuda, int32
import cv2
import sys
import numpy as np
import math
import os
import pathlib
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

IMAGE_PATH = None
try:
    IMAGE_PATH = sys.argv[1]
except IndexError as error: 
    pass

EXPORT_PATH = None

HEIGHT = None
WIDTH = None


@cuda.jit
def cal_non_max_suppression(gradientImage, angleImage, outputImage):
    x,y = cuda.grid(2)
    
    if x < HEIGHT and y < WIDTH:        
        target = gradientImage[x,y]
        angle = angleImage[x,y]
        
        left = 255
        right = 255
        
        #angle 0
        if (-22.5 <= angle <= 22.5) or (157.5 <= angle <= -157.5):
            if y+1 < WIDTH:
                right = gradientImage[x, y+1]
            left = gradientImage[x, y-1]
        #angle 45
        elif (22.5 < angle < 67.5) or (-157.5 < angle < -112.5):
            if x+1 < HEIGHT and y+1 < WIDTH:
                right = gradientImage[x+1, y+1]
            left = gradientImage[x-1, y-1]
        #angle 90
        elif (67.5 <= angle <= 112.5) or (-112.5 <= angle <= -67.5):
            right = gradientImage[x-1, y]
            if x+1 < HEIGHT:
                left = gradientImage[x+1, y]
        #angle 135
        elif (112.5 < angle < 157.5) or (-67.5 < angle < -22.5):
            if y+1 < WIDTH:
                right = gradientImage[x-1, y+1]
            if x+1 < HEIGHT:
                left = gradientImage[x+1, y-1]
            
        if (target >= right) and (target >= left):
            outputImage[x,y] = target
        else:
            outputImage[x,y] = 0
        
   

def make_non_max_suppression(gradientImage, angleImage, exportImage=False):
    if exportImage:
        global EXPORT_PATH
        EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_images")
        try: 
            os.mkdir(EXPORT_PATH) 
        except OSError as error: 
            pass

    global HEIGHT
    global WIDTH
    HEIGHT = gradientImage.shape[0]
    WIDTH = gradientImage.shape[1]

    threadsPerBlock = (32, 32)
    blocks = (HEIGHT//threadsPerBlock[0] + 1, WIDTH//threadsPerBlock[1] + 1)
    
    outputArr = cuda.pinned_array((HEIGHT, WIDTH), dtype=np.int32)
    outputArr.fill(0)
    outputImage = cuda.to_device(outputArr)

    # Berechnung für Non-Maximum Suppresion
    cal_non_max_suppression[blocks, threadsPerBlock](cuda.to_device(gradientImage), cuda.to_device(angleImage), outputImage)
    
    cuda.synchronize()
    
    imageNonMaxSuppression = outputImage.copy_to_host()

    if exportImage:
        imageName = os.path.join(EXPORT_PATH, "NonMaxSuppression.jpg")
        print("[INFO] Exporting image {}".format(imageName))
        cv2.imwrite(imageName, imageNonMaxSuppression)
    
    return imageNonMaxSuppression
    

if __name__ == "__main__":    
    img = cv2.imread(IMAGE_PATH, 0)
    result = Gradient.make_gradient_and_angle(img, True)
    gradient = result[0]
    angle = result[1]
    result = make_non_max_suppression(gradient, angle, True)
    print("Non-Maximum Suppression Result:")
    print(result)    
