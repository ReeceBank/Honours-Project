
from re import I
import sys
import math
import cv2 as cv
import numpy as np

#required installs:
#pip install opencv-python

#skeleton code to help generate hough lines while i work on the:
#k means binarization
#Morphological Closing
#Morphological Erosion
#Morphological Skeleton 
#Morphological Pruning
#Final Hough Transforms

#obersvations:
#v0.3 has issues because of cannying the kmeans

default_file = 'sourceimages/window.png' #tested with julians images

def main():
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(default_file), cv.IMREAD_GRAYSCALE)
    #output canny (not good)
    #easy placeholder until morphological pruning
    dst = cv.Canny(src, 20, 100, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    #kmeans ---------------------
    
    image = cv.imread(default_file, cv.IMREAD_GRAYSCALE) # Loading image
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0) #criteria
    k = 3 # Choosing number of cluster
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS) 

    
    centers = np.uint8(centers) # convert data into 8-bit values 
    print("centres: ",centers)
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
    #quick check to see the minmax of the kmeans (probably an easier way using)
    min = 999
    max = -1
    for n in range(len(segmented_image)):
        if (segmented_image[n][0][0] >= max):
            max = segmented_image[n][0][0]
        elif (segmented_image[n][0][0] <= min):
            min = segmented_image[n][0][0]
    
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > min):
                segmented_image[n][i] = [0,0,0] #the not rows
            elif (segmented_image[n][i][0] == min):
                segmented_image[n][i] = [255,255,255] #the rows (painting them green)
    
    print("min:", min)
    print("max:", max)
    cv.imshow("Kmeans extraction", segmented_image)
    print("First elements: ",segmented_image[0][0])
    print("y: ",len(segmented_image))
    print("x: ",len(segmented_image[0]))

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Change color to RGB (from BGR)
    print("Segmented image: ",segmented_image)
    # kmeans

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main()