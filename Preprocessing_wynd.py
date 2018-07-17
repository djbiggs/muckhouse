##Preprocessing Images to minimize non target edges
## Solving for 1 needs to be rewritten for set based.

import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

def showImageAndWait(name, img):
    WIN_WIDTH = 1280
    imgHeight, imgWidth = img.shape
    winHeight = int(round((WIN_WIDTH / imgWidth) * imgHeight))

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, cv2.resize(img, (WIN_WIDTH, winHeight)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prepImg(imgPath = './competition_files/datasets/train/Image149_2TPP_5R_MT_AS.jpg', isVerbose=True):

    img = cv2.imread(imgPath, 0)

    orig_img = cv2.imread(imgPath, 1)

    img = cv2.Canny(img, threshold1 = 20000, threshold2 = 30000 ,apertureSize = 7)

    ## define the size of the kernel to use for dialation
    ## dilate the image using the kernel
    kernel = np.ones((4,7), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    ##Median Blur Filter
    # uses the median in a 7 by 7 neighborhood to reassign each pixel
    img = cv2.medianBlur(img, ksize = 7)

    ###Gaussian Blur to disturb smaller edges rather than the larger.
    img = cv2.GaussianBlur(src = img, ksize = (3,7), sigmaX = 3)

    # ##Shrinking and enlarging
    # img = cv2.resize(img, None, fx=0.2, fy=0.2)
    # img = cv2.resize(img, None, fx= 5, fy=5)


    ## Or Bilateral filtering for colour http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
    ## cv.GetSize(im)

    # apertureSize argument is the size of the filter for derivative approximation

    kernel2 = np.ones((3,5), np.uint8)
    img = cv2.dilate(img, kernel2, iterations=2)



    # [gray]
    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img


    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 15, -2)


    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [init]

    # [horiz]
    # Specify size on horizontal axis, based on proportion of image px
    cols = horizontal.shape[1]
    horizontal_size = int(cols / (0.3*cols))

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # [vert]
    # Specify size on vertical axis vertical, based on proportion of image px
    rows = vertical.shape[0]
    verticalsize = int(rows / (0.2*rows))
	
	## Resolution 
    res = int(rows * cols)
     
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    img = cv2.addWeighted(horizontal, 1, vertical, 1, 0)

	
    #Contour Search, method takes just the corner coordinates-- need to pass this,creates a numpy list of non redundant contour corner points
#Filters based on  contour areas
    img, contours, hierarchy = cv2.findContours(img, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
     area = cv2.contourArea(contour)
     if area > (res/rows):    
      cv2.drawContours(orig_img, contours, int(cols*0.05), (0,0,255), 2 )
	
    result = img
    #
    # cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    # # HoughLinesP
    # lines = cv2.HoughLinesP(img, 1, math.pi/180.0, 40, np.array([]), 50, 10)
    # a,b,c = lines.shape
    # for i in range(a):
    #     cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
#draws rectangles with a rectangle filter
    (_,contours,_) = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
     area = cv2.contourArea(contour)
     if area > (res/rows):
      (x,y,w,h) = cv2.boundingRect(contour)
      cv2.rectangle(result, (x,y), (x+w,y+h), (255,255,255), 3)
      
     
    if isVerbose:
        showImageAndWait('Final Rectangles', result)
		#cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

    return result
    
if __name__ == '__main__':
    prepImg()