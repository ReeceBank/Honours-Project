import cv2 as cv
import numpy as np
import sys
sys.path.append('./')

#gets drawlines() and drawlinesp()
from versions.linedrawer import drawlines, drawlinesp

#required installs:
#pip install opencv-python


version_name = "v000"

default_file = 'sourceimages/rowtest.png' #tested with julians images
default_parameter_list = []

def run(src_image, parameter_list):
    # Loads an image
    src = cv.imread(cv.samples.findFile(src_image), cv.IMREAD_GRAYSCALE)

    # Copy edges to the images that will display the results in BGR
    csrc = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

    #copys to draw lines on
    srchough = np.copy(csrc)
    srchoughp = np.copy(csrc)

    #draw lines on image - non probabalistic
    lines = cv.HoughLines(src, 1, np.pi / 180, 150, None, 0, 0)
    drawlines(srchough,lines)
    
    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(src, 1, np.pi / 180, 50, None, 50, 10)
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