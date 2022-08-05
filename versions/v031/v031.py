import cv2 as cv
import numpy as np
import sys
sys.path.append('./')

#gets drawlines() and drawlinesp()
from versions.linedrawer import drawlines, drawlinesp

#required installs:
#pip install opencv-python

#an alternate version i found online that was originally used on an elephant
#https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
version_name = "v031"

default_file = 'sourceimages/window.png' #tested with julians images
default_parameter_list = []

def run(src_image, parameter_list):
    # Loads an image
    src = cv.imread(cv.samples.findFile(src_image), cv.IMREAD_ANYCOLOR)
    cv.imshow("Input", src)

    #extract the blue,green,red bands
    b,g,r = cv.split(src)

    #we disregard red and blue
    greensrc = 2*g-r-b
    
    #output canny (not good)
    #easy placeholder until morphological pruning
    greensrc = cv.Canny(greensrc, 280, 290, None, 3)

    #-------------------------------------------alternate
    #at first glance it appears to have the same end results as v030

    # Threshold the image
    ret,img = cv.threshold(greensrc, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(img)==0:
            break

    #-------------------------------------------end
    # Copy edges to the images that will display the results in BGR
    csrc = cv.cvtColor(skel, cv.COLOR_GRAY2BGR)

    #copys to draw lines on
    srchough = np.copy(csrc)
    srchoughp = np.copy(csrc)

    #draw lines on image - non probabalistic
    lines = cv.HoughLines(skel, 1, np.pi / 180, 130, None, 0, 0)
    drawlines(srchough,lines)
    
    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(skel, 1, np.pi / 180, 50, None, 50, 10)
    drawlinesp(srchoughp,linesP)

    #view the green extracted image
    cv.imshow("Source", csrc)
    
    cv.waitKey()
    return 0
    
def importtest():
    return version_name

def main():
    print("Running Directly")
    run(default_file, default_parameter_list)

if __name__ == "__main__":
    main()