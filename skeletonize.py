#!/usr/bin/env python3

import cv2
import sys
import numpy
import os.path

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    # img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1],     # P2,P3,P4,P5
             image[x1][y], image[x1][y_1], image[x][y_1], image[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def fillblanks(image):
    # img = image
    height, width = image.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if (image[i,j] == 0):
                n = neighbours(i,j,image)
                if (numpy.sum(n) > 5):
                    image[i,j] = 1
    return image


def skeletonize(image):
    "the Zhang-Suen Thinning Algorithm"
    image = fillblanks(image)
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
file = os.path.split(filename)
root = file[0]
root = os.path.dirname(root)
file = file[1]
file = os.path.splitext(file)
file = file[0]

mask1 = os.path.join('masks1/', file + '_mask1.png')
mask1 = os.path.join(root, mask1)
mask2 = os.path.join('masks2/', file + '_mask2.png')
mask2 = os.path.join(root, mask2)
mask3 = os.path.join('masks3/', file + '_mask3.png')
mask3 = os.path.join(root, mask3)
mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
mask3 = cv2.imread(mask3, cv2.IMREAD_GRAYSCALE)
mask1points = cv2.findNonZero(mask1)
mask2points = cv2.findNonZero(mask2)
mask3points = cv2.findNonZero(mask3)
mask1rect = cv2.boundingRect(mask1points)
mask2rect = cv2.boundingRect(mask2points)
mask3rect = cv2.boundingRect(mask3points)

segm1 = os.path.join('segm/', file + '_segm1.png')
segm1 = os.path.join(root, segm1)
segm2 = os.path.join('segm/', file + '_segm2.png')
segm2 = os.path.join(root, segm2)
segm3 = os.path.join('segm/', file + '_segm3.png')
segm3 = os.path.join(root, segm3)
segm1 = cv2.imread(segm1, cv2.IMREAD_GRAYSCALE)
segm2 = cv2.imread(segm2, cv2.IMREAD_GRAYSCALE)
segm3 = cv2.imread(segm3, cv2.IMREAD_GRAYSCALE)
segm1 = cv2.bitwise_and(segm1[mask1rect[1] - 20:mask1rect[1] + mask1rect[3] + 20, mask1rect[0] - 20:mask1rect[0] + mask1rect[2] + 20], mask1[mask1rect[1] - 20:mask1rect[1] + mask1rect[3] + 20, mask1rect[0] - 20:mask1rect[0] + mask1rect[2] + 20])
segm2 = cv2.bitwise_and(segm2[mask2rect[1] - 20:mask2rect[1] + mask2rect[3] + 20, mask2rect[0] - 20:mask2rect[0] + mask2rect[2] + 20], mask2[mask2rect[1] - 20:mask2rect[1] + mask2rect[3] + 20, mask2rect[0] - 20:mask2rect[0] + mask2rect[2] + 20])
segm3 = cv2.bitwise_and(segm3[mask3rect[1] - 20:mask3rect[1] + mask3rect[3] + 20, mask3rect[0] - 20:mask3rect[0] + mask3rect[2] + 20], mask3[mask3rect[1] - 20:mask3rect[1] + mask3rect[3] + 20, mask3rect[0] - 20:mask3rect[0] + mask3rect[2] + 20])

skele1 = skeletonize(segm1/255)
print('Skeletonization 1')
skele2 = skeletonize(segm2/255)
print('Skeletonization 2')
skele3 = skeletonize(segm3/255)
print('Skeletonization 3')

skelepath1 = os.path.join('skeleton/', file + '_skele1.png')
skelepath1 = os.path.join(root, skelepath1)
skelepath2 = os.path.join('skeleton/', file + '_skele2.png')
skelepath2 = os.path.join(root, skelepath2)
skelepath3 = os.path.join('skeleton/', file + '_skele3.png')
skelepath3 = os.path.join(root, skelepath3)

cv2.imwrite(skelepath1, skele1*255, (cv2.IMWRITE_PNG_COMPRESSION, 0))
cv2.imwrite(skelepath2, skele2*255, (cv2.IMWRITE_PNG_COMPRESSION, 0))
cv2.imwrite(skelepath3, skele3*255, (cv2.IMWRITE_PNG_COMPRESSION, 0))