##Preprocessing Images to minimize non target edges
## Solving for 1 needs to be rewritten for set based.

import cv2
import numpy as np

img = cv2.imread('test_img_0.jpg', 0)

## define the size of the kernel to use for dialation
## dilate the image using the kernel
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=2)

##Median Blur Filter
# uses the median in a 7 by 7 neighborhood to reassign each pixel
img_dilmed = cv2.medianBlur(img_dilation, ksize = 7)
img_med = cv2.medianBlur(img_dilation, ksize = 7)

##Shrinking and enlarging to do!
## Or Bilateral filtering for colour http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html

# apertureSize argument is the size of the filter for derivative approximation
candm = cv2.Canny(img_dilmed, threshold1 = 0, threshold2 = 50, apertureSize = 3)

## Hough lines feature picking
##cv2.HoughLinesP(edges, rho = 1, theta = math.pi / 180, threshold = 70, minLineLength = 100, maxLineGap = 10)
 
#Contour Search, method takes just the corner coordinates-- need to pass this,creates a numpy list of non redundant contour corner points
candm, contours, hierarchy = cv2.findContours(candm, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
##cnt = contours[4]
##cv2.drawContours(candm, [cnt], -1, (255,0,0), 3)
cv2.drawContours(candm, contours, -1, (100,100,255), 6)
## Display
cv2.imshow('Input', img)

cv2.imshow('Dilation', img_dilation)

cv2.imshow('Dilation Median Blur', img_dilmed)

cv2.imshow('CannyEdge with Fat Contours', candm)


cv2.waitKey(0)





##Gaussian Blur to disturb smaller edges rather than the larger. 
#blur = cv2.GaussianBlur(src = bw, ksize = (3,3), sigma = 0)
## to see the 3 by 3 filter matrix that the image bw is convolved with above:
#cv2.getGaussianKernel(ksize = 3, sigma = 0) * cv2.getGaussianKernel(3, 0).T