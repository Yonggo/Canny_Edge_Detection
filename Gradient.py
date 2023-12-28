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

EXPORT_PATH = None

FILTER_SIZE = 3
IMAGE_PATH = None
try:
    IMAGE_PATH = sys.argv[1]
except IndexError as error: 
    pass

G_X_SOBEL_FILTER = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
G_Y_SOBEL_FILTER = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]], np.int32)

G_HEIGHT = None
G_WIDTH = None


@cuda.jit
def cal_sobel(inputImage, outputImage, filter):
    x,y = cuda.grid(2)
    endPointX = x + FILTER_SIZE #exklusive Stelle
    endPointY = y + FILTER_SIZE #exklusive Stelle
    
    if x < G_HEIGHT and y < G_WIDTH:
        convolution = 0
        for iterX in range(FILTER_SIZE):
            for iterY in range(FILTER_SIZE):
                convolution += inputImage[x+iterX][y+iterY] * filter[iterX][iterY]
                
        outputImage[x,y] = convolution
        
@cuda.jit
def cal_gradient(imgGx, imgGy, outputImage):
    x,y = cuda.grid(2)
    
    if x < G_HEIGHT and y < G_WIDTH:
        outputImage[x,y] = (imgGx[x,y]**2 + imgGy[x,y]**2)**0.5
        
@cuda.jit
def cal_angle(imgGx, imgGy, outputImage):
    x,y = cuda.grid(2)
    
    if x < G_HEIGHT and y < G_WIDTH:
        outputImage[x,y] = math.atan2(imgGy[x,y], imgGx[x,y])*180/3.14
        

def get_length_for_filtered_picture(length):
    count = 0
    while length >= 3:
        count += 1
        length -= 1
        
    return count
   

def make_gradient_and_angle(imageArray, exportImage=False):
    if exportImage:
        global EXPORT_PATH
        EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_images")
        try: 
            os.mkdir(EXPORT_PATH) 
        except OSError as error: 
            pass

    global G_HEIGHT
    global G_WIDTH
    G_HEIGHT = get_length_for_filtered_picture(imageArray.shape[0])
    G_WIDTH = get_length_for_filtered_picture(imageArray.shape[1])

    threadsPerBlock = (32, 32)
    blocks = (G_HEIGHT//threadsPerBlock[0] + 1, G_WIDTH//threadsPerBlock[1] + 1)

    inputImage = cuda.to_device(imageArray)

    # Berechnung für Gx
    outputArrGx = cuda.pinned_array((G_HEIGHT, G_WIDTH), dtype=np.int32)
    outputArrGx.fill(0)
    outputImageGx = cuda.to_device(outputArrGx)

    cal_sobel[blocks, threadsPerBlock](inputImage, outputImageGx, G_X_SOBEL_FILTER)

    # Berechnung für Gy
    outputArrGy = cuda.pinned_array((G_HEIGHT, G_WIDTH), dtype=np.int32)
    outputArrGy.fill(0)
    outputImageGy = cuda.to_device(outputArrGy)

    cal_sobel[blocks, threadsPerBlock](inputImage, outputImageGy, G_Y_SOBEL_FILTER)

    cuda.synchronize()

    imageGx = outputImageGx.copy_to_host()
    imageGy = outputImageGy.copy_to_host()

    if exportImage:
        imageNameGx = os.path.join(EXPORT_PATH, "Gx.jpg")
        imageNameGy = os.path.join(EXPORT_PATH, "Gy.jpg")
        print("[INFO] Exporting image {}".format(imageNameGx))
        cv2.imwrite(imageNameGx, imageGx)
        print("[INFO] Exporting image {}".format(imageNameGy))
        cv2.imwrite(imageNameGy, imageGy)


    # Berechnung für Gradient
    outputArrGradient = cuda.pinned_array((G_HEIGHT, G_WIDTH), dtype=np.int32)
    outputArrGradient.fill(0)
    outputImageGradient = cuda.to_device(outputArrGradient)

    cal_gradient[blocks, threadsPerBlock](imageGx, imageGy, outputImageGradient)

    cuda.synchronize()
    
    imageGradient = outputImageGradient.copy_to_host()
    
    if exportImage:
        imageName = os.path.join(EXPORT_PATH, "Gradient.jpg")
        print("[INFO] Exporting image {}".format(imageName))
        cv2.imwrite(imageName, imageGradient)


    # Berechnung für Arctangent
    outputArrAngle = cuda.pinned_array((G_HEIGHT, G_WIDTH), dtype=np.float32)
    outputArrAngle.fill(0)
    outputImageAngle = cuda.to_device(outputArrAngle)

    cal_angle[blocks, threadsPerBlock](imageGx, imageGy, outputImageAngle)

    cuda.synchronize()
    
    imageAngle = outputImageAngle.copy_to_host()
    
    if exportImage:
        imageName = os.path.join(EXPORT_PATH, "Angle.jpg")
        print("[INFO] Exporting image {}".format(imageName))
        cv2.imwrite(imageName, imageAngle)
    
    result = [imageGradient, imageAngle]
    
    return result
    

if __name__ == "__main__":    
    img = cv2.imread(IMAGE_PATH, 0)
    result = make_gradient_and_angle(img, True)
    gradient = result[0]
    angle = result[1]
    print("Gradient Result:")
    print(gradient)
    print("Angle Result:")
    print(angle)
