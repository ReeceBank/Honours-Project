import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
#external libraries functions
from scipy.stats import circstd, circmean
from skimage.morphology import disk, diamond
#modules
from statistics import stdev, mean
from modules.linedrawer import *

from modules.kmeansbinarization import *
from modules.histogramequalization import *
from modules.colourquantization import *
from modules.morphologicaloperations import *
from modules.skeletonization import *
from modules.cannyedge import *
from modules.greenextract import *
from modules.classifier import *


#required installs:
#pip install opencv-python #core
#pip install matplotlib #graphing
#pip install scipy #stats for stdev
#pip install scikit-image #disk elements

show_image = False
default_file = 'sourceimages/window4.png' #test image
#default_k_value = 2
'''
def kmeans(input_image, k_value):
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
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    #criteria to go to until finished. Set to 100 iterations or 100%.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = k_value # Choosing number of cluster
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
    if show_image:
        cv.imshow("Kmeans extraction", segmented_image)

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Change color to RGB (from BGR) should mess with more options to see results
    #print("Segmented image: ",segmented_image)
    # ------------------ kmeans
    print("Kmeans-Binerization Complete")
    return segmented_image #the kmeans image
'''
'''
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
'''
'''
def colourQuantize(input_image):
    """
    Preprocessing Method. Applies a thresholding colour quantization to a given input image. 
    Returns the colour quantized image.

    @input_image = input image. Works with rgb and non-rgb
    @return: colour quantized image
    """
    
    print("Colour Quantization Started")
    quantize_count = 12 #how many variation to reduce to.

    #safty check that image is an image.
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image
    
    quantized_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 

    if(quantize_count == 12): #default case when you want 12 total channel types (future work)
        for n in range(len(quantized_image)):
            for i in range(len(quantized_image[0])):
                #r band quantized
                if(quantized_image[n][i][0] < 64):
                    quantized_image[n][i][0] = 0
                elif(quantized_image[n][i][0] < 128):
                    quantized_image[n][i][0] = 64
                elif(quantized_image[n][i][0] < 255):
                    quantized_image[n][i][0] = 128
                else:
                    quantized_image[n][i][0] = 255
                
                #g band quantized
                if(quantized_image[n][i][1] < 64):
                    quantized_image[n][i][1] = 0
                elif(quantized_image[n][i][1] < 128):
                    quantized_image[n][i][1] = 64
                elif(quantized_image[n][i][1] < 255):
                    quantized_image[n][i][1] = 128
                else:
                    quantized_image[n][i][1] = 255
                
                #b band quantized
                if(quantized_image[n][i][2] < 64):
                    quantized_image[n][i][2] = 0
                elif(quantized_image[n][i][2] < 128):
                    quantized_image[n][i][2] = 64
                elif(quantized_image[n][i][2] < 255):
                    quantized_image[n][i][2] = 128
                else:
                    quantized_image[n][i][2] = 255
        
    print("Colour Quantization Complete")
    return quantized_image
'''
'''#extracting data functions
def getThetaData(lines):
    """
    Generates a list of line degrees.
    Takes in the lines data generated by the standard Hough Transform. 

    @lines: list of lines generated by opencv's standard hough()
    @return: List of degrees
    """
    #give a lines list from opencv to get a list of slope values
    #for standard hough lines
    rho_data = []
    theta_data = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            rho_data.append(rho)
            theta = lines[i][0][1]
            theta_data.append(theta)

    theta_data = np.arctan(theta_data)
    theta_data = np.degrees(theta_data)

    return theta_data

def getThetaDataP(linesP):
    """
    Generates a list of line degrees.
    Takes in the lines data generated by the probablistic Hough Transform. 

    @linesP: list of lines generated by opencv's probablistic hough()
    @return: List of degrees
    """
    #give a lines list from opencv to get a list of slope values
    #for the probabilistic opencv hough line version.
    theta_datap = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1 = l[0] 
            y1 = l[1] 
            x2 = l[2] 
            y2 = l[3]
            #remove division by zero issue
            if (x2 - x1) == 0:
                thetap = math.inf
            else:
                thetap = (y2 - y1) / (x2 - x1)
            
            theta_datap.append(thetap)

    theta_datap = np.arctan(theta_datap)
    theta_datap = np.degrees(theta_datap)
    #to see all degree values
    #print(theta_datap)

    return theta_datap
'''

'''
def graphTheta(theta_data):
    """
    redundant
    """
    #creates a simple line graph of slope values
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
'''
'''
#central point functions
def findCentrePoints(linesP):
    """
    Find the centre points of each lines. 
    Takes in the lines data generated by the probablistic Hough Transform. 

    @linesP: list of lines generated by opencv's probablistic hough()
    @return: list of centre points in tuple, and a final overall central point.
    """
    #very useful to determin anomolous images as the further from the centre of the image the cetral point is then the more likely its anomalous

    #a list (of tuples) of center point coordinates
    center_points = []
    xcentres_list = []
    ycentres_list = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1 = l[0] 
            y1 = l[1] 
            x2 = l[2] 
            y2 = l[3]

            xcentre = (x1+x2)/2
            xcentres_list.append(xcentre)

            ycentre = (y1+y2)/2
            ycentres_list.append(ycentre)
            centre = (xcentre,ycentre)

            center_points.append(centre)

    #the overall central cluster point of all centre points
    if(len(xcentres_list)>=1): #this comes first since its more common. minor optimization.
        central_point = (sum(xcentres_list)/len(xcentres_list),sum(ycentres_list)/len(ycentres_list))
    else: #safy check for when the image could find lines.
        central_point = (0,0)

    #returns the list of centres and the central cluster point of all lines
    return center_points, central_point

def findCentreDistance(width,height,central_point):
    """
    Finds the distance between the central point of the lines found and the centre of the source image.
    Returns the distance between these two points.

    @width: width of the image
    @height: height of the image
    @central_point of line clusters.

    @return: distance in pixels.
    """
    image_central_point = (width/2,height/2)
    distance = math.dist(image_central_point,central_point)

    return distance

def findCentralAccuracy(width,height,central_point):
    """
    Finds a percentage measure of how close the line cluster central is to the centre of the image.
    with 100% being dead centre and 0% being completely off image.

    
    @width: width of the image
    @height: height of the image
    @central_point of line clusters.

    @return: percentage measure of centre accuracy.
    """
    image_width = width/2
    image_height = height/2
    central_width = central_point[0]
    central_height = central_point[1]
    xoffset = 1 - (math.dist([image_width],[central_width])/image_width)
    xoffset_percentage = xoffset*100
    yoffset = 1 - (math.dist([image_height],[central_height])/image_height)
    yoffset_percentage = yoffset*100

    accuracy_percentage = (xoffset_percentage+yoffset_percentage)/2
    return accuracy_percentage
'''
'''
#morphologyex functions
def MorphSkeleton(image, kernal=None):   
    """
    Refinement Method. Skeletonizes a given image. Returns the skeletonized image.
    Must follow after preprocessing methods. 

    @image: input image to skeletonize.
    @kernal: kernal to use, default to 3x3 cross if none given
    @return: the skeletonized image
    """
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
'''

'''# Anomaly Functions
def AnomalyDecide(accuracy, line_datap, line_min_thres=None,std_dev_thres=None,accuracy_thres=None):
    """
    Classifer used to decide whether an image is anomalous or not. Takes in data about the lines.
    """
    #base cases to judge failure by
    line_count_min = line_min_thres
    if(line_count_min == None):
        line_count_min = 5 #default min line count
    
    line_std_min = std_dev_thres
    if(line_std_min == None):
        line_std_min = 1.0 #default min stdev

    accuracy_min = accuracy_thres
    if(accuracy_min == None):
        accuracy_min = 81 #default min accuracy

    #tests that it failed. For analytics.
    line_count_0_failed = False
    line_count_n0_failed = False
    line_stdev_failed = False
    accuracy_failed = False
    #total tests failed
    failure_count = 0
    #which tests failed
    failed_cases = []

    #complete failure, very likely its anomalous
    if len(line_datap)<1:
        line_count_0_failed = True
        failure_count += 1
    #low line count, may be anomalous
    if len(line_datap)<line_count_min:
        line_count_n0_failed = True
        failure_count += 1

    #low standard deviation, may be anomalous
    if circstd(line_datap)>line_std_min:
        line_stdev_failed = True
        failure_count += 1

    #low accuracy, may be anomalous
    if (accuracy<accuracy_min):
        accuracy_failed = True
        failure_count += 1

    #to further see which tests failed
    failed_cases.append(["line_count_0_passed", line_count_0_failed])
    failed_cases.append(["line_count_n0_passed", line_count_n0_failed])
    failed_cases.append(["line_stdev_passed", line_stdev_failed])
    failed_cases.append(["accuracy_passed", accuracy_failed])

    #boolean operation, if any failed it returns True of if image is anomalous
    return (line_count_0_failed or line_count_n0_failed or line_stdev_failed or accuracy_failed), failure_count, failed_cases
'''
def updateGlobalAccuracy(accuracy_passed_to_me):
    """
    For data analytics. Counts total accuracy. Personal use.
    """
    accuracy_passed_to_me+=1
    return accuracy_passed_to_me

def AnomalyDataCollection(file_to_write_to, image_name, image_height, image_width, accuracy, line_data, line_datap, is_anomalous, failure_count):
    """
    For data analystics. Writes information to a file. Personal use.
    """
    f = open(file_to_write_to, "a")

    f.write(str(image_name)+" ")
    f.write(str(round(accuracy,2))+" ")

    if len(line_datap) >= 1:
        f.write(str(round(circstd(line_datap),2))+" ")
        f.write(str(len(line_datap))+" ")

    else: #if no linedata is present (bad anomaly)
        f.write("0"+" ")
        f.write("0"+" ")

    if(is_anomalous):
        f.write("Anomalous: 1 ")
    else:
        f.write("Nonanomalous: 0 ")
    
    f.write(str(failure_count))

    f.write("\n")
    f.close()
    return 0
'''
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
'''
'''
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
'''
def main(default_k_value, file_to_write, file_accuracy):
    # Loads an image
    src = cv.imread(cv.samples.findFile(default_file))
    srcoriginal = cv.imread(cv.samples.findFile(default_file))
    #src = greenExtract(src)
    #if show_image:
    #    cv.imshow("GreenExtract", src)
    src = histogramEqualization(src)
    if show_image:
        cv.imshow("histogramEqualization", src)
    src = colourQuantize(src)
    if show_image:
        cv.imshow("colourQuantize", src)
    src = kmeans(src,default_k_value)
    if show_image:
        cv.imshow("kmeans", src)


    # A group of kernals
    #kernelrect = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    #kernelcros = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    #kernelline = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    #kerneldiam = diamond(2)
    #kerneldisk = disk(1)

    #double openclose
    src = MorphOpenClose(src)
    src = MorphOpenClose(src)

    #prune
    src = MorphPrune(src)
    if show_image:
        cv.imshow("MorphPrune", src)

    #skeletonization
    src = MorphSkeleton(src)
    if show_image:
        cv.imshow("MorphSkeleton", src)

        
    #src = cannyEdge(src)
    #if show_image:
    #    cv.imshow("CannyEdge", src)
    
    #output canny (not good)
    #easy placeholder until morphological pruning
    #dst = cv.Canny(prune, 20, 100, None, 3)


    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    #draw lines on image - non probabalistic
    #accumulator min dependent on image
    # testing found with big spaced trees = 150, smaller 200x200 windows = 50
    lines = cv.HoughLines(src, 1, np.pi / 180, 100, None, 0, 0)
    drawlines(cdst,lines)
    drawlines(srcoriginal,lines)
    
    #draw lines on image - probabalistic
    #orignally 50/10
    linesP = cv.HoughLinesP(src, 1, np.pi / 180, 50, None, 50, 15)
    drawlinesp(cdstP,linesP)
    drawlinesp(srcoriginal,linesP)

    
    # gets the centre point of the data
    centre_points, central_point = findCentrePoints(linesP)
    print("Centre of the lines: ", central_point)
    image_height = len(src)
    image_width = len(src[0])
    print("Image Height: ", image_height)
    print("Image Width: ", image_width)
    #distance = findCentreDistance(image_width, image_height, central_point)
    #print("Distance from Centre to Central: ", distance)
    accuracy = findCentralAccuracy(image_width, image_height, central_point)
    print("Accuracy of Central: ", str(int(accuracy))+str('%'))

    cdstP = drawlinesCentre(cdstP, central_point)

    if show_image:
        cv.imshow("Original Source", srcoriginal)
        #cv.imshow("Cannied", dst)
        #cv.imshow("Histogramed", histo)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    #if "special" in default_file:
    #    cv.imshow("Special Image", src)


    # looking at some stats
    clean_theta_data = getThetaData(lines)
    clean_theta_dataP = getThetaDataP(linesP)

    #print("Line data: ", clean_theta_data)
    if len(clean_theta_data) >= 2:
        print("Standdev of line data: ", circstd(clean_theta_data))
        print("Mean of line data: ", circmean(clean_theta_data))
        print("Count of line data: ", len(clean_theta_data))
    #print("Line dataP: ", clean_theta_dataP)
    if len(clean_theta_dataP) >= 2:
        print("Standdev of line dataP: ", circstd(clean_theta_dataP))
        print("Mean of line dataP: ", circmean(clean_theta_dataP))
        print("Count of line dataP: ", len(clean_theta_dataP))

    is_image_anomalous, failure_count, failed_tests  = AnomalyDecide(accuracy, clean_theta_dataP)
    AnomalyDataCollection(file_to_write, default_file, image_height, image_width, accuracy, clean_theta_data, clean_theta_dataP, is_image_anomalous, failure_count)

    
    #PH PH PH
    if(is_image_anomalous):
        if "bad" in default_file:
            file_accuracy = updateGlobalAccuracy(file_accuracy)

    if(not is_image_anomalous):
        if "good" in default_file:
            file_accuracy = updateGlobalAccuracy(file_accuracy)
    
    cv.waitKey()
    return file_accuracy


if __name__ == "__main__":
    #simpletest()
    #write the stdev, central poit accuracy, and line counts only to a file for easier mass analysis.
    #also sperate good and bad, and critical failure rate.
    #hand analyzing data is tedious

    #removed a lot of writes
    print("--- Starting ---")
    what_were_testing = "canny_only"

    print("--- Finding Files ---")
    files_to_run = []
    path = "windows/"
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"): 
                files_to_run.append(name)
    
    print("--- Files Found ---")
    for j in range(1):
        file_accuracy = 0
        default_k_value = j+3
        file_to_write_to_global = "Data_k"+str(default_k_value)+"_"+what_were_testing+".txt"
        file_to_write_to_times = "Data_Times_k"+str(default_k_value)+"_"+what_were_testing+".txt"

        f = open(file_to_write_to_global,"w")
        #simple header
        f.write("image_name, central_accuracy, probablistic_stdev, probablistic_count, decision, binary_decision, total_failed_tests \n")
        f.close()
        p = open(file_to_write_to_times,"w")
        p.write("image_name, time\n")
        p.close()
        for i in range(len(files_to_run)):
            default_file = path+str(files_to_run[i])
            print(default_file)
            start = time.time()
            file_accuracy = main(default_k_value, file_to_write_to_global, file_accuracy)
            end = time.time()
            total_time = end - start
            p = open(file_to_write_to_times,"a")
            p.write(str(default_file)+" "+str(total_time)+"\n")
            p.close()

        f = open(file_to_write_to_global,"a")
        print("Global accuracy: ", (file_accuracy/len(files_to_run)))
        f.write(str((file_accuracy/len(files_to_run)))+"\n")
        f.close()
        
    print("--- Finished ---")