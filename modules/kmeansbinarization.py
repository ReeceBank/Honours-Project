#preprocessing module
#for kmeans clustering and binarization

import cv2 as cv
import numpy as np

def kmeans(input_image, k_value=None):
    """
    Preprocessing Method. Applies k-means clustering follower by binarization to a given input image. 
    Returns the binarized image.

    @input_image: input image. Any band.
    @k_value: number of clusters to seperate the pixel data into.
    @return: the binarized image.
    """
    #A kmeans clusters algorithm that takes in an image (ideally grayscale) and applies a binerization to them.
    #kmeans clusters process:
    print("K-means Started") #steps taken from opencv documentation.

    #safty check that image is an image.
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image
    
    image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 
    # reshapethe image into a 2d array to make clustering possible
    pixel_vals = image.reshape((-1,3)) 
    # For supporting cv.kmean
    pixel_vals = np.float32(pixel_vals)

    #criteria to go to until finished. Set to 100 iterations or 100%.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = k_value # Choosing number of cluster
    if (k==None):
        k = 3
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS) 

    
    centers = np.uint8(centers) # convert data into 8-bit values 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
    
    print("K-means Complete")

    #binarization process:
    print("Binerization Started")
    min = 999
    max = -1

    #check the new image for which cluster to binarized.
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > max):
                max = segmented_image[n][i][0]
            elif (segmented_image[n][i][0] < min):
                min = segmented_image[n][i][0]
    
    #binarized the image based on the threshold found.
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > min):
                segmented_image[n][i] = [0,0,0] #the not rows
            elif (segmented_image[n][i][0] <= min):
                segmented_image[n][i] = [255,255,255] #the rows (painting them white)

    print("Binerization Complete")
    
    #if show_image:
    #    cv.imshow("Kmeans extraction", segmented_image)

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Change color to RGB (from BGR) should mess with more options to see results
    #print("Segmented image: ",segmented_image)
    # ------------------ kmeans
    print("Kmeans-Binerization Complete")
    return segmented_image #the kmeans image