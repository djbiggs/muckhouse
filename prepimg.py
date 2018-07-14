#!usr/bin/env python3

import os, fnmatch, cv2


DIR_TRAIN = './competition_files/datasets/train/'
DIR_TEST = './competition_files/datasets/test/'
DIR_OUTPUT = './output/'


def prep(imgPath):
    # do all the things to prep the image
    # return img
    return cv2.imread(imgPath, 0)


def getImages(path = DIR_TRAIN):
    result = []

    for file in os.listdir(path):
        if fnmatch.fnmatch(file.lower(), '*.jpg'):
            result.append(file)

    return result


if __name__ == "__main__":
    listOfImages = getImages(DIR_TRAIN)

    if not(os.path.isdir(DIR_OUTPUT)):
        os.mkdir(DIR_OUTPUT)


    for imgName in listOfImages:
        processedImg = prep(imgName)
        # cv2.imshow(processedImg)
        cv2.imwrite(DIR_OUTPUT + imgName, processedImg)

