import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev, mean
from linedrawer import drawlines, drawlinesp

#required installs:
#pip install opencv-python
#pip install matplotlib

#skeleton code to help generate hough lines while i work on the:
#k means binarization
#Morphological Closing
#Morphological Erosion
#Morphological Skeleton 
#Morphological Pruning
#Final Hough Transforms

#obersvations:
#v0.3 has issues because of cannying the kmeans
#when no quantization: kmeans 6 and above causes issues (no images, too much removed etc)
#with quantization this doesnt occur, k of 12+ still generates WORKING IMAGES, HAZA! i guess.

#default_file = 'sourceimages/window.png' #example with good, but is a png
#default_file = 'sourceimages/bad2-45.png' #example with high standard deviation = bad
#default_file = 'sourceimages/real2.png' #example of real test thats good
default_file = 'sourceimages/real.png' #example of real test thats bad but should be good
#default_file = 'sourceimages/window3.png' #example of real test thats bad but should be good

def kmeans(input_image):
    print("Kmeans")
    #kmeans ---------------------
    
    image = cv.imread(cv.samples.findFile(default_file)) # Loading image
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0) #criteria
    k = 4 # Choosing number of cluster
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS) 

    
    centers = np.uint8(centers) # convert data into 8-bit values 
    print("centres: ",centers)
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
    #quick check to see the minmax of the kmeans (probably an easier way using)
    min = 999
    max = -1
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > max):
                max = segmented_image[n][i][0]
                print("new max found", max)
            elif (segmented_image[n][i][0] < min):
                min = segmented_image[n][i][0]
                print("new min found", min)
    
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > min):
                segmented_image[n][i] = [0,0,0] #the not rows
            elif (segmented_image[n][i][0] <= min):
                segmented_image[n][i] = [255,255,255] #the rows (painting them white)
    
    #print("min:", min)
    #print("max:", max)
    cv.imshow("Kmeans extraction", segmented_image)
    #print("First elements: ",segmented_image[0][0])
    #print("y: ",len(segmented_image))
    #print("x: ",len(segmented_image[0]))

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Change color to RGB (from BGR)
    #print("Segmented image: ",segmented_image)
    # ------------------ kmeans
    return segmented_image #the kmeans image

def colourQuantize(input_image):
    quantize_count = 12
    #take in an image and reduce it down to max of 12 channel types, [0 64 128 255] for r,g,b
    print("Colour Quantization")
    quantized_image = cv.imread(input_image) # Loading image
    quantized_image = cv.cvtColor(quantized_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 

    if(quantize_count == 12): #default case when you want 12 total channel types
        for n in range(len(quantized_image)):
            for i in range(len(quantized_image[0])):
                if(quantized_image[n][i][0] < 64):
                    quantized_image[n][i][0] = 0
                elif(quantized_image[n][i][0] < 128):
                    quantized_image[n][i][0] = 64
                elif(quantized_image[n][i][0] < 255):
                    quantized_image[n][i][0] = 128
                else:
                    quantized_image[n][i][0] = 255
                
                if(quantized_image[n][i][1] < 64):
                    quantized_image[n][i][1] = 0
                elif(quantized_image[n][i][1] < 128):
                    quantized_image[n][i][1] = 64
                elif(quantized_image[n][i][1] < 255):
                    quantized_image[n][i][1] = 128
                else:
                    quantized_image[n][i][1] = 255
                
                if(quantized_image[n][i][2] < 64):
                    quantized_image[n][i][2] = 0
                elif(quantized_image[n][i][2] < 128):
                    quantized_image[n][i][2] = 64
                elif(quantized_image[n][i][2] < 255):
                    quantized_image[n][i][2] = 128
                else:
                    quantized_image[n][i][2] = 255
        
    cv.imshow("Quantized 12", quantized_image)
    return quantized_image

def getThetaData(lines):
    #for standard hough lines
    rho_data = []
    theta_data = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            rho_data.append(rho)
            theta = lines[i][0][1]
            theta_data.append(theta)

    return theta_data

def getThetaDataP(linesP):
    #for the probabilistic opencv hough line version.
    theta_datap = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1 = l[0] 
            y1 = l[1] 
            x2 = l[2] 
            y2 = l[3]
            thetap = (y1 - y2) / (x1 - x2)
            theta_datap.append(thetap)


    return theta_datap

def graphTheta(theta_data):
    theta_data.sort()
    intervals = []
    for i in range(len(theta_data)):
        intervals.append(i)

    # plotting the theta values
    plt.plot(intervals, theta_data)
    # naming the x axis
    plt.xlabel(' Number of Lines ')
    # naming the y axis
    plt.ylabel(' Theta Values ')
    # graph title
    plt.title(' Linegraph showings theta values of lines found by Hough Transform ')

    plt.show()

    return 0

def cleanINFdata(data):
    for i in range(len(data)):
        if data[i]>100:
            data[i] = 100
        elif  data[i]<-100:
            data[i] = -100
    return data

def main():

    prek = colourQuantize(default_file)
    #kmeans ---------------------
    kmean_image = kmeans(prek)
    # ------------------ kmeans
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(default_file))
    #output canny (not good)
    #easy placeholder until morphological pruning
    dst = cv.Canny(kmean_image, 20, 100, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    #draw lines on image - non probabalistic
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    drawlines(cdst,lines)
    
    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    drawlinesp(cdstP,linesP)

    cv.imshow("Original Source", src)
    cv.imshow("Cannied", dst)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    # looking at some stats
    theta_data = getThetaData(lines)
    theta_dataP = getThetaDataP(linesP)

    clean_theta_data = cleanINFdata(theta_data)
    clean_theta_dataP = cleanINFdata(theta_dataP)

    #print("Line data: ", clean_theta_data)
    if len(clean_theta_data) >= 2:
        print("Standdev of line data: ", stdev(clean_theta_data))
        print("Mean of line data: ", mean(clean_theta_data))
    #print("Line dataP: ", clean_theta_dataP)
    if len(clean_theta_dataP) >= 2:
        print("Standdev of line dataP: ", stdev(clean_theta_dataP))
        print("Mean of line dataP: ", mean(clean_theta_dataP))
    graphTheta(theta_dataP)
    
    cv.waitKey()
    return 0
    
def quantizetester():
    test_image = colourQuantize(default_file)
    cv.imshow("output", test_image)
    src = cv.imread(cv.samples.findFile(default_file))
    cv.imshow("input", src)
    cv.waitKey()
    return 0


if __name__ == "__main__":
    main()