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

mask1 = os.path.join('masks1/', mask + '_mask1.png')
mask1 = os.path.join(root, mask1)
mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
roots1 = os.path.join(root, 'ricci/' + mask + '_ricci1.png')
roots1 = cv2.imread(roots1, cv2.IMREAD_GRAYSCALE)

mask2 = os.path.join('masks2/', mask + '_mask2.png')
mask2 = os.path.join(root, mask2)
mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
roots2 = os.path.join(root, 'ricci/' + mask + '_ricci2.png')
roots2 = cv2.imread(roots2, cv2.IMREAD_GRAYSCALE)

mask3 = os.path.join('masks3/', mask + '_mask3.png')
mask3 = os.path.join(root, mask3)
mask3 = cv2.imread(mask3, cv2.IMREAD_GRAYSCALE)
roots3 = os.path.join(root, 'ricci/' + mask + '_ricci3.png')
roots3 = cv2.imread(roots3, cv2.IMREAD_GRAYSCALE)

gt = os.path.join(root, 'gt2-segm/' + mask + '_gtmask1.png')
gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)

gt1 = cv2.bitwise_and(gt, mask1)
gt2 = cv2.bitwise_and(gt, mask2)
gt3 = cv2.bitwise_and(gt, mask3)

union1 = cv2.bitwise_or(gt1, roots1)
intersection1 = cv2.bitwise_and(gt1, roots1)
union2 = cv2.bitwise_or(gt2, roots2)
intersection2 = cv2.bitwise_and(gt2, roots2)
union3 = cv2.bitwise_or(gt3, roots3)
intersection3 = cv2.bitwise_and(gt3, roots3)

jaccard1 = numpy.sum(intersection1)/numpy.sum(union1)
dice1 = (2*jaccard1)/(jaccard1 + 1)
jaccard2 = numpy.sum(intersection2)/numpy.sum(union2)
dice2 = (2*jaccard2)/(jaccard2 + 1)
jaccard3 = numpy.sum(intersection3)/numpy.sum(union3)
dice3 = (2*jaccard3)/(jaccard3 + 1)

print("Dice Mask1: ", dice1)
print("Jaccard Mask1: ", jaccard1)
print("Dice Mask2: ", dice2)
print("Jaccard Mask2: ", jaccard2)
print("Dice Mask3: ", dice3)
print("Jaccard Mask3: ", jaccard3)

cv2.waitKey()
