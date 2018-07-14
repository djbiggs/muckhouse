import cv2
import numpy as np 
import glob


### A script for the base case of cropping a core box ie by finding the centres of hough line intersection clusters... 
### seems to work ok for ultra base cases, 
### TO DO refine the preprocessing using??? marker-based image segmentation using watershed algorithm
### Write for sets.

def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

##images = [cv2.imread(file) for file in glob.glob("/Users/davidbiggs/Desktop/Core/*.jpg")]
img = cv2.imread("test_img_4.jpg" )

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
img = cv2.Canny(img, threshold1 = 10000, threshold2 = 30000 ,apertureSize = 7)

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

# run the Hough transform
lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=100, maxLineGap=20, minLineLength=100)

# segment the lines
delta = 10
h_lines, v_lines = segment_lines(lines, delta)

# draw the segmented lines
houghimg = img.copy()
for line in h_lines:
    for x1, y1, x2, y2 in line:
        color = [255,0,0] # color hoz lines red
        cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=2)
for line in v_lines:
    for x1, y1, x2, y2 in line:
        color = [0,0,255] # color vert lines blue
        cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=2)

cv2.imshow("Segmented Hough Lines", houghimg)
cv2.waitKey(0)
cv2.imwrite('hough.png', houghimg)

# find the line intersection points
Px = []
Py = []
for h_line in h_lines:
    for v_line in v_lines:
        px, py = find_intersection(h_line, v_line)
        Px.append(px)
        Py.append(py)

# draw the intersection points
intersectsimg = img.copy()
for cx, cy in zip(Px, Py):
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    color = np.random.randint(0,255,3).tolist() # random colors
    cv2.circle(intersectsimg, (cx, cy), radius=2, color=color, thickness=-1) # -1: filled circle

cv2.imshow("Intersections", intersectsimg)
cv2.waitKey(0)
cv2.imwrite('intersections.png', intersectsimg)

# use clustering to find the centers of the data clusters
P = np.float32(np.column_stack((Px, Py)))
nclusters = 4
centers = cluster_points(P, nclusters)
print(centers)

# draw the center of the clusters
for cx, cy in centers:
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    cv2.circle(img, (cx, cy), radius=20, color=[255,255,255], thickness=-1) # -1: filled circle


cv2.imshow("Center of intersection clusters",img)
cv2.waitKey(0)
cv2.imwrite('corners.png', img	)