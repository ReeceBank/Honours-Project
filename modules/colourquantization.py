#preprocessing module
#for colour quantization

import cv2 as cv

def colourQuantize(input_image):
    """
    Preprocessing Method. Applies a thresholding colour quantization to a given input image. 
    Returns the colour quantized image.

    @input_image = input image. Works with rgb and non-rgb
    @return: colour quantized image
    """
    
    print("Colour Quantization Started")
    quantize_count = 12 #how many variation to reduce to.

    #safty check that image is an image.
    if isinstance(input_image, str):
        input_image = cv.imread(input_image) # Loading image
    
    quantized_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 

    if(quantize_count == 12): #default case when you want 12 total channel types (future work)
        for n in range(len(quantized_image)):
            for i in range(len(quantized_image[0])):
                #r band quantized
                if(quantized_image[n][i][0] < 64):
                    quantized_image[n][i][0] = 0
                elif(quantized_image[n][i][0] < 128):
                    quantized_image[n][i][0] = 64
                elif(quantized_image[n][i][0] < 255):
                    quantized_image[n][i][0] = 128
                else:
                    quantized_image[n][i][0] = 255
                
                #g band quantized
                if(quantized_image[n][i][1] < 64):
                    quantized_image[n][i][1] = 0
                elif(quantized_image[n][i][1] < 128):
                    quantized_image[n][i][1] = 64
                elif(quantized_image[n][i][1] < 255):
                    quantized_image[n][i][1] = 128
                else:
                    quantized_image[n][i][1] = 255
                
                #b band quantized
                if(quantized_image[n][i][2] < 64):
                    quantized_image[n][i][2] = 0
                elif(quantized_image[n][i][2] < 128):
                    quantized_image[n][i][2] = 64
                elif(quantized_image[n][i][2] < 255):
                    quantized_image[n][i][2] = 128
                else:
                    quantized_image[n][i][2] = 255
        
    print("Colour Quantization Complete")
    return quantized_image