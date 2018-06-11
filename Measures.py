#!/usr/bin/env python3

import cv2
import sys
import numpy
import math
import os.path

filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
height, width = img.shape
mask = os.path.split(filename)
root = mask[0]
root = os.path.dirname(root)
mask = mask[1]
mask = os.path.splitext(mask)
mask = mask[0]
mask2 = os.path.join('masks2/', mask + '_mask2.png')
mask2 = os.path.join(root, mask2)
mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
roots2 = os.path.join(root, 'ricci/' + mask + '_ricci2.png')
roots2 = cv2.imread(roots2, cv2.IMREAD_GRAYSCALE)
cv2.imshow('roots2', roots2)
gt = os.path.join(root, 'gt2-segm/' + mask + '_gtmask1.png')
gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
gt2 = cv2.bitwise_and(gt, mask2)
cv2.imshow('gt2', gt2)

union2 = cv2.bitwise_or(gt2, roots2)
intersection2 = cv2.bitwise_and(gt2, roots2)

jaccard = numpy.sum(intersection2)/numpy.sum(union2)
dice = (2*jaccard)/(jaccard + 1)

print(dice)
print(jaccard)

cv2.waitKey()
