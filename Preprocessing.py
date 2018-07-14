"""
Preprocesses images to minimize non target edges.
"""

import cv2
import numpy as np
import math


img = cv2.imread('test_img_6.jpg', 0)


cv2.imshow('Input', img)
cv2.waitKey(0)

## define the size of the kernel to use for dialation
## dilate the image using the kernel
kernel = np.ones((5,9), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Dilation', img)
cv2.waitKey(0)

##Median Blur Filter
# uses the median in a 7 by 7 neighborhood to reassign each pixel
img = cv2.medianBlur(img, ksize = 7)

cv2.imshow('Dilation Median Blur', img)
cv2.waitKey(0)

###Gaussian Blur to disturb smaller edges rather than the larger.
##img = cv2.GaussianBlur(src = img, ksize = (3,3), sigmaX = 3)

#cv2.imshow('GBlur', img)
#cv2.waitKey(0)

##Shrinking and enlarging
img = cv2.resize(img, None, fx=0.2, fy=0.2)
img = cv2.resize(img, None, fx= 5, fy=5)

cv2.imshow('Resized', img)
cv2.waitKey(0)

## Or Bilateral filtering for colour http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
## cv.GetSize(im)

# apertureSize argument is the size of the filter for derivative approximation
img = cv2.Canny(img, threshold1 = 20000, threshold2 = 50000 ,apertureSize = 7)

cv2.imshow('CannyEdge', img)
cv2.waitKey(0)

kernel2 = np.ones((2,2), np.uint8)
img = cv2.dilate(img, kernel2, iterations=2)

cv2.imshow('Dialation2', img)
cv2.waitKey(0)

## Hough lines feature picking
##hlines =cv2.HoughLinesP(candm, rho = 1, theta = math.pi/180, threshold = 70, minLineLength = 100, maxLineGap = 10)

#Contour Search, method takes just the corner coordinates-- need to pass this,creates a numpy list of non redundant contour corner points
img, contours, hierarchy = cv2.findContours(img, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, 5, (100,100,255), 2)

cv2.imshow('FinalContours', img)

## Display

##cv2.imshow('Hough Lines', hlines)


cv2.waitKey(0)


