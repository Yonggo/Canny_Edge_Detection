import Gradient
import NonMaxSuppression
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

MAX_WEIGHT = 0.5
MIN_WEIGHT = 0.025

MAX = 255*MAX_WEIGHT
MIN = 255*MIN_WEIGHT


@cuda.jit
def cal_double_threshold(nonMaxSupImage, outputImage):
    x,y = cuda.grid(2)
    
    if x < HEIGHT and y < WIDTH:        
        target = nonMaxSupImage[x,y]        
        if target > MAX:
            outputImage[x,y] = 255
        elif target < MIN:
            outputImage[x,y] = 0
        else:
            outputImage[x,y] = target
        
   

def make_double_threshold(nonMaxSupImage, exportImage=False):
    if exportImage:
        global EXPORT_PATH
        EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_images")
        try: 
            os.mkdir(EXPORT_PATH) 
        except OSError as error: 
            pass

    global HEIGHT
    global WIDTH
    HEIGHT = nonMaxSupImage.shape[0]
    WIDTH = nonMaxSupImage.shape[1]

    threadsPerBlock = (32, 32)
    blocks = (HEIGHT//threadsPerBlock[0] + 1, WIDTH//threadsPerBlock[1] + 1)
    
    outputArr = cuda.pinned_array((HEIGHT, WIDTH), dtype=np.int32)
    outputArr.fill(0)
    outputImage = cuda.to_device(outputArr)

    # Berechnung für Double Threadshold
    cal_double_threshold[blocks, threadsPerBlock](cuda.to_device(nonMaxSupImage), outputImage)
    
    cuda.synchronize()
    
    imageDoubleThreashold = outputImage.copy_to_host()

    if exportImage:
        imageName = os.path.join(EXPORT_PATH, "DoubleThreashold.jpg")
        print("[INFO] Exporting image {}".format(imageName))
        cv2.imwrite(imageName, imageDoubleThreashold)
    
    return imageDoubleThreashold
    

if __name__ == "__main__":    
    img = cv2.imread(IMAGE_PATH, 0)
    result = Gradient.make_gradient_and_angle(img, True)
    gradient = result[0]
    angle = result[1]
    nonMaxSupImage = NonMaxSuppression.make_non_max_suppression(gradient, angle, True)
    result = make_double_threshold(nonMaxSupImage, True)
    print("Double Threashold Result:")
    print(result)  