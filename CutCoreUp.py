#!usr/bin/env python3
"""
Prep Image
Batch process 10 or more images with the preprocessing steps.
Reads from competition source folder and writes optimised images to output directory.

Usage:
  prepimg.py [--source DIR] [--dest DIR] [--no-limit] [-v]
  prepimg.py (-h | --help)

Options:
  -h --help         Show this screen.
  -v --verbose      Give verbose output, with images to tap through.
  --no-limit        Preps all images in source folder.
  --dest DEST       Specify an destination directory (optional).
  --source SRC      Specify a source directory (optional).
"""

import cv2
import fnmatch
import os

from docopt import docopt
from HalfImage import NotCore

DIR_TRAIN = './competition_files/datasets/train/'
DIR_TEST = './competition_files/datasets/test/'
DIR_OUTPUT = './nocore/'
LIMIT_COUNT = 10
NO_LIMIT_PARTY = """
🌟 🎆 🌟 🎇 🌟 🎆 🌟 🎇 🌟 🎆 🌟 🎇 🌟 🎆 🌟 🎇

Lemme hear ya say yeah (wow)
Lemme hear ya say yeah (wow)

No, no, no, no, no, no, no, no, no ,no ,no, no there's no limit
No, no, no ,no, no, no, no, no, no, no, no, no there's no limit
No no limits, we'll reach for the sky
No valley to deep, no mountain too high
No no limits, we'll reach for the sky
We do what we want and we do it with pride.

🎇 🌟 🎆 🌟 🎇 🌟 🎆 🌟 🎇 🌟 🎆 🌟 🎇 🌟 🎆 🌟
"""


def getImages(srcDir):
    result = []

    for file in os.listdir(srcDir):
        if fnmatch.fnmatch(file.lower(), '*.jpg'):
            result.append(file)

    return result


if __name__ == "__main__":
    args = docopt(__doc__)

    # Set vars according to args
    isVerbose = args['--verbose']
    isLimited = not(args['--no-limit'])

    if(not(isLimited)):
        print(NO_LIMIT_PARTY)

    if (args['--source'] == None):
        srcDir = DIR_TRAIN
    else:
        srcDir = string(args['SRC'])

    if (args['--dest'] == None):
        destDir = DIR_OUTPUT
    else:
        destDir = string(args['DEST'])

    print('Prepping images...')

    # Create dest dir
    if os.path.isdir(destDir):
        print('🗂 ' + destDir + ' already exists.')
    else:
        print('🗂 ' + destDir + ' doesn\'t exist, creating it.')
        os.mkdir(destDir)

    # Get images from src dir
    print('🔍 Getting the list of images from ' + srcDir)
    listOfImages = getImages(srcDir)

    # Prep the images and write them to disk
    if len(listOfImages) >= 0:
        num = 0

        for imgName in listOfImages:
            if (isLimited and (num >= LIMIT_COUNT)):
                break

            num += 1
            print('🖼 ' + str(num) + ': ' + imgName)
            print('processing...')
            processedImg = NotCore(srcDir + imgName, isVerbose)
            print('writing...')
            cv2.imwrite(DIR_OUTPUT + imgName, processedImg)
    else:
        print('Couldn\'t find any images is source directory.')

    # Done
    print('')
    print('🎉 Done! {0} images prepped.'.format(num))

    if isLimited:
        print('Use the --no-limit flag to prep all images.')
