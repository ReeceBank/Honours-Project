#preprocessing module
#for green extraction of rgb images.

import cv2 as cv
import numpy as np

def greenExtract(in_image):
    """
    Side Preprocessing Method. Applies the green extract method to a given image. Must be an RGB image band (or BGR).

    @in_image: input image in 3 channel band with middle channel being green.
    @return: green extracted image.
    """
    #safty check that image is an image.
    image = in_image
    if isinstance(image, str):
        image = cv.imread(image) # Loading image

    #extract the blue,green,red bands
    greensrc = np.copy(image)
    #we disregard red and blue
    for n in range(len(greensrc)):
        for i in range(len(greensrc[0])):
            greensrc[n][i][0] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
            greensrc[n][i][1] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
            greensrc[n][i][2] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
    
    return greensrc