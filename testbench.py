import sys
import math
import cv2 as cv
import numpy as np
import os
import time

#external libraries functions
from scipy.stats import circstd, circmean

#modules
from modules.linedrawer import * # for line drawing
from modules.kmeansbinarization import *
from modules.histogramequalization import *
from modules.colourquantization import *
from modules.morphologicaloperations import *
from modules.skeletonization import *
from modules.cannyedge import *
from modules.greenextract import *
from modules.classifier import * #for classification


#required installs:
#pip install opencv-python #core
#pip install matplotlib #graphing
#pip install scipy #stats for stdev
#pip install scikit-image #disk elements

show_image = False
default_file = 'sourceimages/window4.png' #test image

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
    cv.destroyAllWindows()
    return file_accuracy


if __name__ == "__main__":
    #simpletest()
    #write the stdev, central poit accuracy, and line counts only to a file for easier mass analysis.
    #also sperate good and bad, and critical failure rate.
    #hand analyzing data is tedious

    #removed a lot of writes
    print("--- Starting ---")
    what_were_testing = "bestcase"

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