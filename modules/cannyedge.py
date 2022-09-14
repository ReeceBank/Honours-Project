#refinement module
#for canny edge detection
import cv2 as cv
import numpy as np

def cannyEdge(in_image, in_kernal = None, lowerband_in=None, upperband_in=None):
    """
    Refinement Method. Applies the canny edge detector algorithm to an image to highlight edges. Ideally in the binarized image.
    
    @in_image: input image to canny
    @in_kernal: input kernal to use for guassian blur. default to 5x5 rect if none specificed.
    @lowerband_in: lowerband to use for canny. default to 20 if non specified.
    @Upperband_in: upperband to use for canny. default to 100 if non specified.

    @return: canny edged image.
    """
    #check image
    image = in_image
    if isinstance(image, str):
        image = cv.imread(image) # Loading image

    #kernal to use for guassian blur
    gauss_kernal = in_kernal
    if(gauss_kernal == None):
        gauss_kernal = (5,5)
    
    #lowerband of canny
    lowerband = lowerband_in
    if(lowerband==None):
        lowerband = 20

    #upperband of canny
    upperband = upperband_in
    if(upperband==None):
        upperband = 100

    blured_image = cv.GaussianBlur(image,gauss_kernal,0)
    canny_image = cv.Canny(blured_image,lowerband,upperband)

    return canny_image