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

default_file = 'testcase/bad_13_300.png' #tested with julians images
default_parameter_list = []

def run(src_image, parameter_list):
    # Loads an image
    src = cv.imread(cv.samples.findFile(src_image), cv.IMREAD_ANYCOLOR)
    cv.imshow("Input", src)
    print(src)

    #extract the blue,green,red bands
    greensrc = np.copy(src)
    #we disregard red and blue
    for n in range(len(greensrc)):
        for i in range(len(greensrc[0])):
            greensrc[n][i][0] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
            greensrc[n][i][1] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
            greensrc[n][i][2] = max(2*greensrc[n][i][1]-greensrc[n][i][0]-greensrc[n][i][2],0)
    
    cv.imshow("greensrc", greensrc)
    cv.waitKey()

    
    cv.waitKey()
    return 0
    
def importtest():
    return version_name

def main():
    print("Running Directly")
    run(default_file, default_parameter_list)

if __name__ == "__main__":
    main()