import Gradient
import NonMaxSuppression
import DoubleThreshold
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
def cal_strong_edge_tracking_hysteresis(image):
    x,y = cuda.grid(2)
    strong = 255 
    if x < HEIGHT and y < WIDTH and image[x,y] == strong:
        iterRange = cuda.local.array(3, int32)
        iterRange[0] = -1
        iterRange[1] = 0
        iterRange[2] = 1
        for iX in iterRange:
            if 0 <= x+iX < HEIGHT:
                for iY in iterRange:
                    if 0 <= y+iY < WIDTH:
                        if iX != 0 and iY != 0:
                            if 0 < image[x+iX, y+iY] < strong:
                                image[x+iX, y+iY] = strong

                              
@cuda.jit
def cal_weak_edge_tracking_hysteresis(image):
    x,y = cuda.grid(2)
    strong = 255 
    if (x < HEIGHT and y < WIDTH) and (0 < image[x,y] < strong):       
        iterRange = cuda.local.array(3, int32)
        iterRange[0] = -1
        iterRange[1] = 0
        iterRange[2] = 1
        for iX in iterRange:
            if 0 <= x+iX < HEIGHT:
                for iY in iterRange:
                    if 0 <= y+iY < WIDTH:
                        if iX != 0 and iY != 0:
                            if image[x+iX, y+iY] == strong:
                                image[x,y] = strong
                                break
                                
                                
@cuda.jit
def clean_up(image):
    x,y = cuda.grid(2)
    strong = 255 
    if x < HEIGHT and y < WIDTH:       
        if 0 < image[x,y] < 255:
            image[x,y] = 0
                                
                                
def cal_edge_tracking_hysteresis_by_multiple_time(blocks, threadsPerBlock, image, iterNr):
    for iter in range(iterNr):
        #cal_strong_edge_tracking_hysteresis[blocks, threadsPerBlock](image)
        #cuda.synchronize()
        cal_weak_edge_tracking_hysteresis[blocks, threadsPerBlock](image)
        cuda.synchronize()
   

def make_edge_tracking_hysteresis(doubleThresholdImage, exportImage=False):
    if exportImage:
        global EXPORT_PATH
        EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_images")
        try: 
            os.mkdir(EXPORT_PATH) 
        except OSError as error: 
            pass

    global HEIGHT
    global WIDTH
    HEIGHT = doubleThresholdImage.shape[0]
    WIDTH = doubleThresholdImage.shape[1]

    threadsPerBlock = (32, 32)
    blocks = (HEIGHT//threadsPerBlock[0] + 1, WIDTH//threadsPerBlock[1] + 1)
    
    image = cuda.to_device(doubleThresholdImage)

    # Berechnung für Edge Tracking by Hysteresis
    cal_edge_tracking_hysteresis_by_multiple_time(blocks, threadsPerBlock, image, 100)
    
    clean_up[blocks, threadsPerBlock](image)
    
    cuda.synchronize()
    
    imageEdgeTackingHysteresis = image.copy_to_host()

    if exportImage:
        imageName = os.path.join(EXPORT_PATH, "EdgeTackingHysteresis.jpg")
        print("[INFO] Exporting image {}".format(imageName))
        cv2.imwrite(imageName, imageEdgeTackingHysteresis)
    
    return imageEdgeTackingHysteresis
    

if __name__ == "__main__":    
    img = cv2.imread(IMAGE_PATH, 0)
    result = Gradient.make_gradient_and_angle(img, True)
    gradient = result[0]
    angle = result[1]
    nonMaxSupImage = NonMaxSuppression.make_non_max_suppression(gradient, angle, True)
    doubleThresholdImage = DoubleThreshold.make_double_threshold(nonMaxSupImage, True)
    result = make_edge_tracking_hysteresis(doubleThresholdImage, True)
    print("Edge Tracking Hysteresis Result:")
    print(result)  