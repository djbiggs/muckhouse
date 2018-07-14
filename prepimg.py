#!usr/bin/env python3
"""
Batch process source images with the preprocessing steps.
Writes optimised images to output directory.
"""

import cv2
import fnmatch
import os
from Preprocessing import prepImg


DIR_TRAIN = './competition_files/datasets/train/'
DIR_TEST = './competition_files/datasets/test/'
DIR_OUTPUT = './output/'


def getImages(path = DIR_TRAIN):
    result = []

    for file in os.listdir(path):
        if fnmatch.fnmatch(file.lower(), '*.jpg'):
            result.append(file)

    return result


if __name__ == "__main__":
    path = DIR_TRAIN

    print('Getting the list of images from ' + path)
    listOfImages = getImages(path)

    if not(os.path.isdir(DIR_OUTPUT)):
        print(DIR_OUTPUT + ' already exists.')
        os.mkdir(DIR_OUTPUT)
    else:
        print(DIR_OUTPUT + ' doesn\'t exist, creating it.')

    num = 0
    for imgName in listOfImages:
        num += 1
        print(str(num) + ': ' + imgName)
        print('processing...')
        processedImg = prepImg(path + imgName, False)
        print('writing...')
        cv2.imwrite(DIR_OUTPUT + imgName, processedImg)

    print('')
    print('Done!')
