#refinement module
#for skeletonization.
#nb credit to: https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331

import cv2 as cv
import numpy as np

def MorphSkeleton(image, kernal=None):   
    '''
    Refinement Method. Skeletonizes a given image. Returns the skeletonized image.
    Must follow after preprocessing methods. 

    @image: input image to skeletonize.
    @kernal: kernal to use, default to 3x3 cross if none given
    @return: the skeletonized image
    '''    
    print("Skeletonizing Started")
    #check that input is image 
    if isinstance(image, str):
        image = cv.imread(image) # Loading image
    
    skel = np.zeros(image.shape, np.uint8)
    # Use a cross Kernel by default.
    element = kernal
    if(element == None):
        element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

    #repeat until image has been skeletonized
    #not own implimentation. Common way of doing it but taken from:
    #https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
    while True:
        open = cv.morphologyEx(image, cv.MORPH_OPEN, element)
        temp = cv.subtract(image, open)
        eroded = cv.erode(image, element) 
        skel = cv.bitwise_or(skel,temp)
        image = eroded.copy()
        #if image has been completely eroded, stop
        if cv.countNonZero(image)==0:
            break
    
    print("Skeletonizing Complete")
    return skel