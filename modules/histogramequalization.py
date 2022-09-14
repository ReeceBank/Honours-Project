#preprocessing module
#for histogram equaltization

import cv2 as cv

def histogramEqualization(input_image):
    """
    Preprocessing Method. Applies Histogram Equalization to a given input image. 
    Returns the histogramed image.

    @input_image: input image of ideally grayscale.
    @return: Histogram equalized image.
    """
    #A histogram equalization function that takes in an input image and equalizes the colour variance. 
    #Recommended by Patrick to fix the issue where some tif have heavy shading due to cliffs/clouds.
    #Ideally takes in a grayscale image.
    print("Histogram Equalization Started")

    #safty check that image is an image.
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image

    #grayscale anyway, safty first.
    c_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    histo_image = cv.equalizeHist(c_image) 

    print("Histogram Equalization Complete")
    return histo_image