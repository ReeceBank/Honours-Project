#classifier methods
#collects data based on supplied data and decides if an image is anomalous or not.
import math
import numpy as np
from scipy.stats import circstd, circmean

# ------- raw extracting data functions --------
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

# ------- finding functions --------
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

# ------- classifying functions --------
def AnomalyDecide(found_accuracy, found_line_datap, line_min_thres=None,std_dev_thres=None,accuracy_thres=None):
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
    if len(found_line_datap)<1:
        line_count_0_failed = True
        failure_count += 1
    #low line count, may be anomalous
    if len(found_line_datap)<line_count_min:
        line_count_n0_failed = True
        failure_count += 1

    #low standard deviation, may be anomalous
    if circstd(found_line_datap)>line_std_min:
        line_stdev_failed = True
        failure_count += 1

    #low accuracy, may be anomalous
    if (found_accuracy<accuracy_min):
        accuracy_failed = True
        failure_count += 1

    #to further see which tests failed
    failed_cases.append(["line_count_0_passed", line_count_0_failed])
    failed_cases.append(["line_count_n0_passed", line_count_n0_failed])
    failed_cases.append(["line_stdev_passed", line_stdev_failed])
    failed_cases.append(["accuracy_passed", accuracy_failed])

    #boolean operation, if any failed it returns True of if image is anomalous
    return (line_count_0_failed or line_count_n0_failed or line_stdev_failed or accuracy_failed), failure_count, failed_cases