import cv2 as cv
import numpy as np
import sys
sys.path.append('./')

#gets drawlines() and drawlinesp()
from versions.linedrawer import drawlines, drawlinesp

#required installs:
#pip install opencv-python


version_name = "v011"

default_file = 'sourceimages/rowtest.png' #tested with julians images

#can feed in your own parameter_list, or for later to use a ML algorithm
default_canny_list = [40, 130, None, 3]
default_hough_list = [1, np.pi / 180, 150, None, 0, 0]
default_houghp_list = [1, np.pi / 180, 50, None, 50, 10]
default_parameter_list = []

default_parameter_list.append(default_canny_list)
default_parameter_list.append(default_hough_list)
default_parameter_list.append(default_houghp_list)

def run(src_image, parameter_list):
    # Loads an image
    src = cv.imread(cv.samples.findFile(src_image), cv.IMREAD_GRAYSCALE)
    
    #output canny (not good)
    #easy placeholder until morphological pruning
    cannysrc = cv.Canny(src, #input image
    parameter_list[0][0], #threshold 1 (min)
    parameter_list[0][1], #threshold 2 (max)
    parameter_list[0][2], #output edges (none since we dont really care)
    parameter_list[0][3]) #appeture size

    # Copy edges to the images that will display the results in BGR
    colourcannysrc = cv.cvtColor(cannysrc, cv.COLOR_GRAY2BGR)

    #copys to draw lines on
    srchough = np.copy(colourcannysrc)
    srchoughp = np.copy(colourcannysrc)

    #draw lines on image - non probabalistic
    lines = cv.HoughLines(cannysrc, #input image
    parameter_list[1][0], #rho
    parameter_list[1][1], #theta (in radians)
    parameter_list[1][2], #threshold
    parameter_list[1][3], #output edges (none since lines variable instead)
    parameter_list[1][4], #snr = dont use (leave 0 unless advanced)
    parameter_list[1][5]) #stn = dont use
    drawlines(srchough,lines)
    
    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(cannysrc, #input image
    parameter_list[2][0], #rho
    parameter_list[2][1], #theta (in radians)
    parameter_list[2][2], #threshold
    parameter_list[2][3], #output edges (none since lines variable instead)
    parameter_list[2][4], #min thres
    parameter_list[2][5]) #max thres
    drawlinesp(srchoughp,linesP)

    #view the original image used
    cv.imshow("Source", src)
    
    cv.waitKey()
    return 0
    
def importtest():
    return version_name

def main():
    print("Running Directly")
    run(default_file, default_parameter_list)

if __name__ == "__main__":
    main()