import cv2
import numpy as np 

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

img = cv2.imread('test_img_0.jpg')

# preprocessing
img = cv2.resize(img, None, fx=.5, fy=.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
dilated = cv2.dilate(edges, np.ones((4,4), dtype=np.uint8))

cv2.imshow("Dilated", dilated)
cv2.waitKey(0)
cv2.imwrite('dilated.png', dilated)

# run the Hough transform
lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=100, maxLineGap=20, minLineLength=50)

# segment the lines
delta = 10
h_lines, v_lines = segment_lines(lines, delta)

# draw the segmented lines
houghimg = img.copy()
for line in h_lines:
    for x1, y1, x2, y2 in line:
        color = [0,0,255] # color hoz lines red
        cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
for line in v_lines:
    for x1, y1, x2, y2 in line:
        color = [255,0,0] # color vert lines blue
        cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)

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
    cv2.circle(img, (cx, cy), radius=4, color=[0,0,255], thickness=-1) # -1: filled circle

cv2.imshow("Center of intersection clusters", img)
cv2.waitKey(0)
cv2.imwrite('corners.png', img)