#!/usr/bin/env python3

import cv2
import sys
import numpy
import math
import skeletonize as sk
import os.path

filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
height, width = img.shape
skel = sk.skeletonize(img/255)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(numpy.uint8(skel))
label_hue = numpy.uint8(179*labels/numpy.max(labels))
blank_ch = 255*numpy.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

cv2.imshow('labeled.png', labeled_img)
# for i in range(width):
#     for j in range(heigh):
#         if (image[i,j] == )
cv2.waitKey()
