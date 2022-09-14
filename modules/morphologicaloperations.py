#refinement method
#morphological operations for open, close, and pruning.

import cv2 as cv
from skimage.morphology import disk

def MorphOpenClose(image, kernal_open=None,kernal_close=None, iterations_open=None, iterations_close=None):
    """
    Refinement Method. Applies a opening followed by a closing morphological operation to an image.

    @image: input openclose.
    @kernal_open: kernal to use for opening. default to disk kernal if none specified.
    @kernal_close: kernal to use for closing. default to 4x4 rect kernal if none specified.
    @iterations_open: number of iterations to apply opening. default to 1 if none specified
    @iterations_close: number of iterations to apply closing. default to 1 if none specified
    
    @retun: the openclosed morph-operated image.
    """
    #check that input is image 
    if isinstance(image, str):
        image = cv.imread(image) # Loading image

    element_open = kernal_open
    if(element_open == None):
        element_open = disk(1)

    element_close = kernal_close
    if(element_close == None):
        element_close = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))

    
    ito = iterations_open
    if(ito == None):
        ito = 1

    itc = iterations_close
    if(itc == None):
        itc = 1
    
    #kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, element_open, ito)

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, element_close, itc) #closing = dilate and then erode
    
    return closing

def MorphPrune(image, kernal_erode=None,kernal_prune=None, iterations_erode=None, iterations_prune=None):
    """
    Refinement Method. Applies a pruning operation to an image, perferably after morphological operations and before skeletonization.
    Applies a erosion followed by an opening.

    @image: input image to prune.
    @kernal_erode: kernal to use for erosion. default to 2x2 rect kernal if none specified.
    @kernal_prune: kernal to use for opening. default to 'line' kernal if none specified.
    @iterations_erode: number of iterations to apply eroison. default to 1 if none specified
    @iterations_prune: number of iterations to apply opening. default to 1 if none specified
    
    @retuns the pruned image.
    """
    if isinstance(image, str):
        image = cv.imread(image) # Loading image
    
    element_open = kernal_erode
    if(element_open == None):
        element_open = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

    element_close = kernal_prune
    if(element_close == None):
        element_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))

    
    ite = iterations_erode
    if(ite == None):
        ite = 1

    itp = iterations_prune
    if(itp == None):
        itp = 1
    
    #performe an erosion operation
    erosion = cv.morphologyEx(image, cv.MORPH_ERODE, element_open, ite)

    #perform a pruning operation
    prune = cv.morphologyEx(erosion, cv.MORPH_CLOSE, element_close, itp)

    return prune

def MorphExFull(in_image):
    """
    Refinement Method. Takes in a given input image and applies two iterations of open-closing, followed by one iteration of pruning.

    @in_image: input image.
    @return: Fully morphologically operated image.
    """
    image = in_image
    if isinstance(in_image, str):
        image = cv.imread(image) # Loading image

    #double openclose
    image = MorphOpenClose(image)
    image = MorphOpenClose(image)

    #prune
    image = MorphPrune(image)

    return image