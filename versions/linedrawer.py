#to draw lines on a processed image
import math
import cv2 as cv

draw_colour = (0,0,255) # bgr in opencv
central_colour = (255,0,0) # central point colour
line_thickness = 2

version_name = "Linedrawer"

def drawlines(image, line_list):
    #draws the lines onto the (presumibly) processed image
    if line_list is not None:
        for i in range(0, len(line_list)):
            rho = line_list[i][0][0]
            theta = line_list[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) #(x1,y1)
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a))) #(x2,y2)
            cv.line(image, pt1, pt2, draw_colour, line_thickness, cv.LINE_AA)
    
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", image)
    return image

def drawlinesp(image, probabilistic_line_list):
    #draws the lines of the probabilistic lines onto the processed image
    #data appears to be given by open cv mirrored and with respect to the horizontal (0)
    #or not mirrored and with respect to the -horizontal (180)
    if probabilistic_line_list is not None:
        for i in range(0, len(probabilistic_line_list)):
            l = probabilistic_line_list[i][0]
            cv.line(image, (l[0], l[1]), (l[2], l[3]), draw_colour, line_thickness, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", image)
    return image

def drawlinesCentre(image, central_point):
    #draws a 10x10 marker on the found central point of the image
    cv.line(image, ((int(central_point[0])+5), int(central_point[1])), ((int(central_point[0])-5), int(central_point[1])), central_colour, line_thickness+1, cv.LINE_AA)
    cv.line(image, (int(central_point[0]), (int(central_point[1])+5)), (int(central_point[0]), (int(central_point[1])-5)), central_colour, line_thickness+1, cv.LINE_AA)
    return image


def importtest():
    return version_name

def main():
    print("Running", version_name)

if __name__ == "__main__":
    main()