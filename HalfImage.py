
import cv2
import math
import numpy as np


## Crops the images for non full core trays to create a haar classifier training set of not a core row. 

def NotCore(imgPath = './competition_files/datasets/train/Image149_2TPP_5R_MT_AS.jpg', isVerbose=True):

    img = cv2.imread(imgPath, 0)
    cv2.imshow("cropped", img)
    #cv2.waitKey(0)
    
    ymin = int(img.shape[1]/2)
    xmin = int(img.shape[0]/2) 
    
      
    result = img[0:ymin, 0:xmin]
    #cv2.imshow("cropped", result)
    #cv2.waitKey(0)
    

    return result
    
if __name__ == '__main__':
    NotCore()
    