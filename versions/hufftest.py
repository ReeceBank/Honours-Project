import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#external libraries functions
from scipy.stats import circstd, circmean
from skimage.morphology import disk, diamond
#modules
from statistics import stdev, mean
from linedrawer import drawlines, drawlinesp, drawlinesCentre


#required installs:
#pip install opencv-python #core
#pip install matplotlib #graphing
#pip install scipy #stats for stdev
#pip install scikit-image #disk elements

#skeleton code to help generate hough lines while i work on the:
#Morphological Closing
#Morphological Erosion
#Morphological Skeleton 
#Morphological Pruning

#obersvations:
#v0.3 has issues because of cannying the kmeans
#when no quantization: kmeans 6 and above causes issues (no images, too much removed etc)
#with quantization this doesnt occur, k of 12+ still generates WORKING IMAGES, HAZA! i guess.
#the ideal image would always be at 45 degrees. anomalous images have high stdev (0.7) and clean have low (0.08)

    #notes on closing: with window.png, 1 iteration, 5x5 rect
    # Standard Lines
    # stdev from 0 to 0 (lol)
    # mean from 72.25 to 72.25 (expected)
    # linecount from 19 to 22 (amazing!)
    # Probablistic Lines
    # stdev from 0.5508 to 0.5334 (better)
    # mean from 86.624 to 88.911 (better)
    # linecount from 75 to 64 (argubly worse?)


#default_file = 'sourceimages/window.png' #example with good, but is a png so cant use

#default_file = 'sourceimages/bad2-45.png' #example with high standard deviation = bad ( high stdev )
#default_file = 'sourceimages/bad2.png' #example where being horizontal (or vertical) messes with results. ( low stdev )
#default_file = 'sourceimages/real2.png' #example of real test thats good ( low stdev )

#default_file = 'sourceimages/real.png' #example of real test thats bad but should be good ( with shadows )
#default_file = 'sourceimages/window3.png' #example of real test ( horrible spaced trees )

#default_file = 'sourceimages/small.png' #example of real test ( horrible spaced trees ) but windowed, so good.

#default_file = 'sourceimages/mess.png' #is anomaly, says its not based on stdev, line count too low = anomoly
#default_file = 'sourceimages/bent.png' #example of real test thats bad but should be good ( with shadows )

default_file = 'sourceimages/bad4.png' #test image



def kmeans(input_image):
    #A kmeans clusters algorithm that takes in an image (ideally grayscale) and applies a binerization to them.
    print("Kmeans")
    #kmeans ---------------------

    #for when the passed image is actually the name not the array of the image (ie when called first in a modular situation)
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image
    
    image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0) #criteria
    k = 3 # Choosing number of cluster
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS) 

    
    centers = np.uint8(centers) # convert data into 8-bit values 
    #print("centres: ",centers)
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape)) # reshape data into the original image dimensions
    
    print("Kmeans Complete")
    #quick check to see the minmax of the kmeans (probably an easier way using)
    print("Binerization")
    min = 999
    max = -1
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > max):
                max = segmented_image[n][i][0]
                #print("new max found", max)
            elif (segmented_image[n][i][0] < min):
                min = segmented_image[n][i][0]
                #print("new min found", min)
    
    for n in range(len(segmented_image)):
        for i in range(len(segmented_image[0])):
            if (segmented_image[n][i][0] > min):
                segmented_image[n][i] = [0,0,0] #the not rows
            elif (segmented_image[n][i][0] <= min):
                segmented_image[n][i] = [255,255,255] #the rows (painting them white)
    print("Binerization Complete")
    
    cv.imshow("Kmeans extraction", segmented_image)

    segmented_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2GRAY) # Change color to RGB (from BGR) should mess with more options to see results
    #print("Segmented image: ",segmented_image)
    # ------------------ kmeans
    print("Kmeans-Binerization Complete")
    return segmented_image #the kmeans image

def histogramEqualization(input_image):
    #A histogram equalization function that takes in an input image and equalizes the colour variance. 
    #Recommended by Patrick to fix the issue where some tif have heavy shading due to cliffs/clouds.
    #Ideally takes in a grayscale image.
    print("Histogram Equalization")

    #for when the passed image is actually the name not the array of the image (ie when called first in a modular situation)
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image

    c_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) # convert to grayscale
    histo_image = cv.equalizeHist(c_image)
    print("Histogram Equalization Complete")
    return histo_image

def colourQuantize(input_image):
    #reduces the colour spectrum from 255*255*255 colour variations to 3*3*3 total variations
    #ideally takes in NON grayscale image, RGB only. (or 3 channel images)
    quantize_count = 12 #how many variation to reduce to. static value because i planned to try out more than 12 combinations.
    #take in an image and reduce it down to max of 12 channel types, [0 64 128 255] for r,g,b
    print("Colour Quantization")

    #for when the passed image is actually the name not the array of the image (ie when called first in a modular situation)
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image
    
    quantized_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 

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
    print("Colour Quantization Complete")
    return quantized_image

def getThetaData(lines):
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

def graphTheta(theta_data):
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

def cleanINFdata(data):
    #simple fix to inf data when determining slope
    # for i in range(len(data)):
    #     if data[i]>100:
    #         data[i] = 100
    #     elif  data[i]<-100:
    #         data[i] = -100
    return data

def findCentrePoints(linesP):
    #finds the centre points of the probablistic hough transform lines
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
    if(len(xcentres_list)>=1):
        central_point = (sum(xcentres_list)/len(xcentres_list),sum(ycentres_list)/len(ycentres_list))
    else:
        central_point = (-1,-1)

    #returns the list of centres and the central cluster point of all lines
    return center_points, central_point

def findCentreDistance(width,height,central_point):
    image_central_point = (width/2,height/2)
    distance = math.dist(image_central_point,central_point)

    return distance

def findCentralAccuracy(width,height,central_point):
    #a percentage mesure of how close the cluster centre is to the centre of the image
    #with 100% being dead centre and 0% being completely off image.
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

def MorphSkeleton():
    return 0

    
def MorphOpenClose():
    return 0

def MorphPrune():
    return 0 

def AnomalyDecide(accuracy, line_data, line_datap):
    accuracy_decision = ""
    if accuracy < 80:
        accuracy_decision = "medium"



    return 0

def main():
    # Loads an image
    src = cv.imread(cv.samples.findFile(default_file))
    histo = histogramEqualization(src)
    prek = colourQuantize(histo)
    kmean_image = kmeans(prek)


    # --------- MORPHOLOGICAL EXPERIMENT ----------
    kernelrect = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    #print("Rect:","\n", kernelrect)
    kerneldisk = disk(1)
    #print("Disk:","\n", kerneldisk)
    kernelcros = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    #print("Cross:","\n", kernelcros)
    kerneldiam = diamond(2)
    #print("Diamond:","\n", kerneldiam)
    kernelline = cv.getStructuringElement(cv.MORPH_RECT, (4, 1))
    #print("Line:","\n", kernelline)

    #kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    opening = cv.morphologyEx(kmean_image, cv.MORPH_OPEN, kerneldisk, iterations=1)
    cv.imshow("Opening", opening)

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernelrect, iterations=1) #closing = dilate and then erode
    cv.imshow("Closing", closing)

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening2 = cv.morphologyEx(closing, cv.MORPH_OPEN, kerneldisk, iterations=1)
    cv.imshow("Opening2", opening2)

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    closing2 = cv.morphologyEx(opening2, cv.MORPH_CLOSE, kernelrect, iterations=1) #closing = dilate and then erode
    cv.imshow("Closing2", closing2)
    
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erosion = cv.morphologyEx(closing2, cv.MORPH_ERODE, kernelrect, iterations=1)
    cv.imshow("Erosion", erosion)

    erosion = closing2

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    #opening3 = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel, iterations=1)
    #cv.imshow("Opening3", opening3)

    # --------- SKELETONIZATION EXPERIMENT ----------
    #ret,img = cv.threshold(erosion, 127, 255, 0)

    # Step 1: Create an empty skeleton
    #size = np.size(img)
    skel = np.zeros(erosion.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv.morphologyEx(erosion, cv.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv.subtract(erosion, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(erosion, element)
        skel = cv.bitwise_or(skel,temp)
        erosion = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(erosion)==0:
            break

    cv.imshow("Skeleton", skel)

    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    prune = cv.morphologyEx(skel, cv.MORPH_CLOSE, kernelline, iterations=2)
    
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    #prune = cv.morphologyEx(prune, cv.MORPH_ERODE, kernel, iterations=1)
    cv.imshow("Prune", prune)

    
    #output canny (not good)
    #easy placeholder until morphological pruning
    dst = cv.Canny(prune, 20, 100, None, 3)


    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(prune, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    #draw lines on image - non probabalistic
    #accumulator min dependent on image
    # testing found with big spaced trees = 150, smaller 200x200 windows = 50
    lines = cv.HoughLines(prune, 1, np.pi / 180, 100, None, 0, 0)
    drawlines(cdst,lines)
    drawlines(src,lines)
    
    #draw lines on image - probabalistic
    #orignally 50/10
    linesP = cv.HoughLinesP(skel, 1, np.pi / 180, 50, None, 50, 15)
    drawlinesp(cdstP,linesP)
    drawlinesp(src,linesP)

    
    # gets the centre point of the data
    centre_points, central_point = findCentrePoints(linesP)
    print("Centre of the lines: ", central_point)
    image_height = len(src)
    image_width = len(src[0])
    print("Image Height: ", image_height)
    print("Image Width: ", image_width)
    distance = findCentreDistance(image_width, image_height, central_point)
    print("Distance from Centre to Central: ", distance)
    accuracy = findCentralAccuracy(image_width, image_height, central_point)
    print("Accuracy of Central: ", str(int(accuracy))+str('%'))

    cdstP = drawlinesCentre(cdstP, central_point)

    cv.imshow("Original Source", src)
    cv.imshow("Cannied", dst)
    cv.imshow("Histogramed", histo)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    # looking at some stats
    theta_data = getThetaData(lines)
    theta_dataP = getThetaDataP(linesP)

    clean_theta_data = cleanINFdata(theta_data)
    clean_theta_dataP = cleanINFdata(theta_dataP)

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
    #graphTheta(theta_dataP)

    anomaly_decision, reason_decision = AnomalyDecide(accuracy, clean_theta_data, clean_theta_dataP)
    
    cv.waitKey()
    return 0
    
def simpletest():
    test_image = histogramEqualization(default_file)
    #cv.imshow("output", test_image)
    #src = cv.imread(cv.samples.findFile(default_file))
    #cv.imshow("input", src)
    #cv.waitKey()
    return 0


if __name__ == "__main__":
    #simpletest()
    main()