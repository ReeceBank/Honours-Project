#Switch board to switch between versions
from ast import Break
from tkinter import E
import cv2 as cv

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

#modules to check
#HE CE CQ GE KM LD MO SK
def modularCase(image_name, show_image,useHE,useCQ,useGE,useMorphs,useSkele,useCanny):
    try:
        src = cv.imread(cv.samples.findFile(str("windows/"+image_name+".png")))
    except:
        print("File not found:",str("windows/"+image_name+".png"))
        return None

    src = cv.imread(cv.samples.findFile(str("windows/"+image_name+".png")))
    srcoriginal = np.copy(src)
    #Green extraction
    if(useGE):
        src = greenExtract(src)
        if show_image:
            cv.imshow("greenExtract", src)

    #histogram equalization
    if(useHE):
        src = histogramEqualization(src)
        if show_image:
            cv.imshow("histogram equalization", src)
    
    #colour quantization
    if(useCQ):
        src = colourQuantize(src)
        if show_image:
            cv.imshow("colour quantization", src)

    #kmeans-binarization
    src = kmeans(src,k_value=3)
    if show_image:
        cv.imshow("kmeans-binarization", src)

    #morphological operations
    if(useMorphs):
        src = MorphExFull(src)
        if show_image:
            cv.imshow("Morph Full", src)

    #skeletonization
    if(useSkele):
        src = MorphSkeleton(src)
        if show_image:
            cv.imshow("Morph Skeleton", src)

    #Canny
    if(useCanny):
        src = cannyEdge(src)
        if show_image:
            cv.imshow("Canny Edge", src)

    # Copy images to draw lines onto
    cdstP = np.copy(srcoriginal)

    #draw lines on image - probabalistic
    linesP = cv.HoughLinesP(src, 1, np.pi / 180, 50, None, 50, 15)
    drawlinesp(cdstP,linesP)

    # gets the centre point of the data
    centre_points, central_point = findCentrePoints(linesP)
    accuracy = findCentralAccuracy(len(src[0]), len(src), central_point)

    drawlinesCentre(cdstP, central_point)

    if show_image:
        cv.imshow("Original Source", srcoriginal)
        cv.imshow("Detected Lines (in red) - Probabilistic Hough Line Transform", cdstP)


    # looking at some stats
    clean_theta_dataP = getThetaDataP(linesP)
    #classifier running
    is_image_anomalous, failure_count, failed_tests  = AnomalyDecide(accuracy, clean_theta_dataP)

    if(is_image_anomalous): #display what the classifier thinks
        print("Image",image_name,"is anomalous.")
        print("Image",image_name,"failed",failure_count,"tests.")
        guessAnomalyType(failed_tests)
    else:
        print("Image",image_name,"is not anomalous.")
    
    return 0


def allcase(image_name, show_image=False):    # Loads an image
    try:
        src = cv.imread(cv.samples.findFile(str("windows/"+image_name+".png")))
    except: #will spit out a opencv warning but thats out of my control.
        print("File not found:",str("windows/"+image_name+".png"))
        return None

    src = cv.imread(cv.samples.findFile(str("windows/"+image_name+".png")))
    srcoriginal = np.copy(src)
    #histogram equalization
    src = histogramEqualization(src)
    if show_image:
        cv.imshow("histogramEqualization", src)
    
    #colour quantization
    src = colourQuantize(src)
    if show_image:
        cv.imshow("colourQuantize", src)

    #kmeans-binarization
    src = kmeans(src,k_value=3)
    if show_image:
        cv.imshow("kmeans", src)

    #morphological operations
    src = MorphExFull(src)
    if show_image:
        cv.imshow("Morph", src)

    #skeletonization
    src = MorphSkeleton(src)
    if show_image:
        cv.imshow("MorphSkeleton", src)

    # Copy images to draw lines onto
    cdstP = np.copy(srcoriginal)

    #draw lines on image - probabalistic
    #orignally 50/10
    linesP = cv.HoughLinesP(src, 1, np.pi / 180, 50, None, 50, 15)
    drawlinesp(cdstP,linesP)

    # gets the centre point of the data
    centre_points, central_point = findCentrePoints(linesP)
    accuracy = findCentralAccuracy(len(src[0]), len(src), central_point)

    drawlinesCentre(cdstP, central_point)

    if show_image:
        cv.imshow("Original Source", srcoriginal)
        cv.imshow("Detected Lines (in red) - Probabilistic Hough Line Transform", cdstP)


    # looking at some stats
    clean_theta_dataP = getThetaDataP(linesP)

    #classifier running
    is_image_anomalous, failure_count, failed_tests  = AnomalyDecide(accuracy, clean_theta_dataP)

    if(is_image_anomalous): #display what the classifier thinks
        print("Image",image_name,"is anomalous.")
        print("Image",image_name,"failed",failure_count,"tests.")
        guessAnomalyType(failed_tests)
    else:
        print("Image",image_name,"is not anomalous.")
    
    return 0

if __name__ == "__main__":
    user_input = True
    while(user_input):
        code = input('Enter function to run\n"best": to run best module combinations\n"default": to use default test case\n"modular": to choose own modules to use\n"q": to quit\n')
        if code == 'best':
            file_name = input('Enter file name\n')
            show_images = input('Do you want to see the image results? (y/n)\n')
            if(show_images == 'y'):
                show_images = True
            else:
                show_images = False
        
            allcase(file_name,show_images)

        elif code == '' or code == 'default': #simple default
            default_file = 'good_1_300'
            show_images = True
            allcase(default_file, show_images)

        elif code == 'modular':
            file_name = input('Enter file name\n')

            # show the images produced
            show_image = input('Show final images? (y/n)\n')
            if (show_image == 'y'):
                show_image = True
            else:
                show_image = False

            #use Histogram Equalization
            useHE = input('Use Histogram Equalization? (y/n)\n')
            if (useHE == 'y'):
                useHE = True
            else:
                useHE = False

            #use Colour Quantization
            useCQ = input('Use Colour Quantization? (y/n)\n')
            if (useCQ == 'y'):
                useCQ = True
            else:
                useCQ = False

            #use Green Extract
            useGE = input('Use Green Extract? (y/n) (not advised)\n')
            if (useGE == 'y'):
                useGE = True
            else:
                useGE = False

            #use Morphological Operations
            useMorphs = input('Use Morphological Operations? (y/n)\n')
            if (useMorphs == 'y'):
                useMorphs = True
            else:
                useMorphs = False

            #use Skeletonization
            useSkele = input('Use Skeletonization? (y/n)\n')
            if (useSkele == 'y'):
                useSkele = True
            else:
                useSkele = False

            #use Canny Edge Detector
            useCanny = input('Use Canny Edge Detector? (y/n)\n')
            if (useCanny == 'y'):
                useCanny = True
            else:
                useCanny = False

            modularCase(file_name, show_image, useHE,useCQ,useGE,useMorphs,useSkele,useCanny)

        elif code == 'q':
            user_input = False
            
        cv.waitKey()
        cv.destroyAllWindows()