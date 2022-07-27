from PIL import Image
import numpy
print("Simple script to see .tif modes")
#how this works:
#run it, ala 
# > python3 fileviewer
#or in an IDE
#default image name here
imagename = 'messed.tif'
#for when you just hit 'enter' without entering a file name


def openimage(name):
    print("Opening:",name)
    image = Image.open(name)
    #number of frames in the image (hidden sub images basically)
    print("Total frames:",image.n_frames)

    #bool check for when you want to show each frame, and not just their mode/width/height/frame count etc
    displaycheck = input("Do you wish to show() each frame? (y/n)")

    for i in range(image.n_frames):
        image.seek(i)
        print("Frame",i,"is mode:",image.mode)#so we can see the mode

        #can return this, or send it off for processing
        #contains all the data of each pixel
        imagedata = numpy.array(image)

        #dont want to process showing 20+ images... 
        if(displaycheck == 'y'):
                image.show() #shows the image

        #same deal as before, dont stall forever loading images arrays.
        if(displaycheck == 'y'):
                #display the array, will mostly be white, alpha 0 pixels since its the edges too it seems. (for default images)
                print(imagedata)

        #image resolution attributes
        print("Frame",i,"width is",len(imagedata[0]))
        print("Frame",i,"height is",len(imagedata))
        #alternate ways to see width/height:
        #print(image.width)
        #print(image.height)

        #from here you can work on your stuff alone, use imagedata array to do the window etc.
        #this is as far as im going to go until Jason explains how to get rgb channels for f mode 'messed up' images.

        #also refer to julians file reader for further step details. (like transperancy)

        '''
        PIL pixel formats:

        RGB -   24bits per pixel, 8-bit-per-channel RGB), 3 channels
        RGBA -  (8-bit-per-channel RGBA), 4 channels
        RGBa -  (8-bit-per-channel RGBA, remultiplied alpha), 4 channels
        1 -     1bpp, often for masks, 1 channel
        L -     8bpp, grayscale, 1 channel
        P -     8bpp, paletted, 1 channel
        I -     32-bit integers, grayscale, 1 channel
        F -     32-bit floats, grayscale, 1 channel
        CMYK -  8 bits per channel, 4 channels
        YCbCr-  8 bits per channel, 3 channels
        '''

def main():
    name = input('enter image name (without .tif):')
    if name == '':
        openimage(imagename)
    else:
        try:
            openimage(name+'.tif')
        except:
            print("file not found")



if __name__=="__main__":
    while(True):
        main()