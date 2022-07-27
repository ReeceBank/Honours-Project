import cv2 as cv
import numpy as np
import sys
sys.path.append('./')

#gets drawlines() and drawlinesp()
from versions.linedrawer import drawlines, drawlinesp

#required installs:
#pip install opencv-python


version_name = "v020"

default_file = 'sourceimages/rowtest.png' #tested with julians images
default_parameter_list = []

def run(src_image, parameter_list):
    # Loads an image
    src = cv.imread(cv.samples.findFile(src_image), cv.IMREAD_ANYCOLOR)

    #extract the blue,green,red bands
    b,g,r = cv.split(src)

    #we disregard red and blue
    greensrc = 2*g-r-b
    
    #output canny (not good)
    #easy placeholder until morphological pruning
    dst = cv.Canny(greensrc, 280, 290, None, 3)

    # Copy edges to the images that will display the results in BGR
    csrc = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    #copys to draw lines on
    srchough = np.copy(csrc)
    srchoughp = np.copy(csrc)

    #draw lines on image - non probabalistic
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    drawlines(srchough,lines)
    
    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    drawlinesp(srchoughp,linesP)

    #view the green extracted image
    cv.imshow("Source", greensrc)
    
    cv.waitKey()
    return 0
    
def importtest():
    return version_name

def main():
    print("Running Directly")
    run(default_file, default_parameter_list)

if __name__ == "__main__":
    main()