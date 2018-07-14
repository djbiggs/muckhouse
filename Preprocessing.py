#!usr/bin/env python3
"""
Preprocesses image functions to minimize non target edges.
"""

import cv2
import math
import numpy as np


def showImageAndWait(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def prepImg(imgPath = './competition_files/datasets/train/Image199_2TPP_5R_MT_AS.jpg', isVerbose = True):

    # Read in the image
    result = cv2.imread(imgPath, 0)

    if isVerbose:
        showImageAndWait('Input', result)

    # Dilate
    # Define the size of the kernel for dialation, then dilate.
    kernel = np.ones((5,9), np.uint8)
    result = cv2.dilate(result, kernel, iterations=1)

    if isVerbose:
        showImageAndWait('Dilation', result)


    # Median Blur Filter
    # uses the median in a 7 by 7 neighborhood to reassign each pixel
    result = cv2.medianBlur(result, ksize = 7)

    if isVerbose:
        showImageAndWait('Dilation', result)


    # Gaussian Blur to disturb smaller edges rather than the larger.
    # result = cv2.GaussianBlur(src = img, ksize = (3,3), sigmaX = 3)
    # if isVerbose:
    #     showImageAndWait('GBlur', result)


    # Shrinking and enlarging
    result = cv2.resize(result, None, fx=0.2, fy=0.2)
    result = cv2.resize(result, None, fx= 5, fy=5)

    if isVerbose:
        showImageAndWait('Resized', result)


    # Or Bilateral filtering for colour
    # http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
    # cv.GetSize(im)

    # Canny edge detection
    # apertureSize is size of filter for derivative approximation
    result = cv2.Canny(result, threshold1 = 20000, threshold2 = 50000, apertureSize = 7)

    if isVerbose:
        showImageAndWait('Canny Edge', result)


    # Dilate again
    kernel2 = np.ones((2,2), np.uint8)
    result = cv2.dilate(result, kernel2, iterations = 2)

    cv2.imshow('Dialation 2', result)


    # Hough lines feature picking
    # hlines = cv2.HoughLinesP(candm, rho = 1, theta = math.pi/180, threshold = 70, minLineLength = 100, maxLineGap = 10)
    # if isVerbose:
    #     showImageAndWait('Hough Lines', result)


    # Contour Search,
    # method takes corner coordinates-- need to pass this,creates a numpy list of non redundant contour corner points
    result, contours, hierarchy = cv2.findContours(result, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, 5, (100,100,255), 2)

    if isVerbose:
        showImageAndWait('Final Contours', result)

    return result


if __name__ == '__main__':
    prepImg()

