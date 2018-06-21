#!/usr/bin/env python3

import cv2
import sys
import numpy
import math
import os.path

def getLine(windowWidth, angle):
    line = numpy.zeros((windowWidth,windowWidth))
    c = int(numpy.floor(windowWidth/2))
    angle = angle/180*numpy.pi
    p1 = [windowWidth*numpy.cos(angle), windowWidth*numpy.sin(angle)]
    p2 = p1[:]
    p2[:] = [-1*x for x in p2]
    p1[0] = int(numpy.ceil(p1[0]) + c)
    p2[0] = int(numpy.ceil(p2[0]) + c)
    p1[1] = int(-numpy.ceil(p1[1]) + c)
    p2[1] = int(-numpy.ceil(p2[1]) + c)
    cv2.line(line, (c,c), tuple(p1), 1, 1)
    cv2.line(line, (c,c), tuple(p2), 1, 1)
    return line

def ricci(image, windowRange):
    height, width = image.shape
    output = numpy.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            if (image[i,j] != 0):
                lineAverageLists = list()
                windowAverageList = list()
                for k in range(windowRange[0], windowRange[1] + 2, 2):
                    winWidth = int(numpy.floor(k/2))
                    window = image[i - winWidth:i + winWidth + 1, j - winWidth:j + winWidth + 1]
                    windowAverageList.append(numpy.mean(window))
                    lineAverage = list()
                    for l in range(0, 180, 15):
                        line = getLine(k, l)
                        temp = cv2.bitwise_and(window, line)
                        sum = numpy.sum(temp)
                        lineAverage.append(sum/winWidth)
                    lineAverageLists.append(lineAverage)
                winnerAnglesPerWindow = numpy.argmax(lineAverageLists, axis=1)
                winnerGreyPerWindow = list()
                for l in range(int((windowRange[1]-windowRange[0])/2)):
                    winnerGreyPerWindow.append(lineAverageLists[l][winnerAnglesPerWindow[l]])
                winner = numpy.argmax(winnerGreyPerWindow)
                intensity = winnerGreyPerWindow[winner] - windowAverageList[winner]
                winWidth = (winner + 1)*2 + 1
                winnerLine = getLine(winWidth , winnerAnglesPerWindow[winner]*15)*intensity
                winWidth = int(numpy.floor(winWidth/2))
                output[i - winWidth:i + winWidth + 1, j - winWidth:j + winWidth + 1] = numpy.add(output[i - winWidth:i + winWidth + 1, j - winWidth:j + winWidth + 1], winnerLine)
    return output


def eraseObjects(img, npixel):
    objects = numpy.zeros(img.shape, numpy.uint8)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if (len(contours[i]) < npixel):
            for j in range(len(contours[i])):
                x = contours[i][j][0][0]
                y = contours[i][j][0][1]
                cv2.circle(objects, (x, y), 1, 255, -1)
    img2 = img - objects
    return img2


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
mask2 = os.path.join('masks2/', mask + '_mask2.png')
mask2 = os.path.join(root, mask2)
mask3 = os.path.join('masks3/', mask + '_mask3.png')
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
# plant1 = cv2.bitwise_and(img[mask1rect[1] - 20:mask1rect[1] + mask1rect[3] + 20, mask1rect[0] - 20:mask1rect[0] + mask1rect[2] + 20], mask1[mask1rect[1] - 20:mask1rect[1] + mask1rect[3] + 20, mask1rect[0] - 20:mask1rect[0] + mask1rect[2] + 20])
# plant2 = cv2.bitwise_and(img[mask2rect[1] - 20:mask2rect[1] + mask2rect[3] + 20, mask2rect[0] - 20:mask2rect[0] + mask2rect[2] + 20], mask2[mask2rect[1] - 20:mask2rect[1] + mask2rect[3] + 20, mask2rect[0] - 20:mask2rect[0] + mask2rect[2] + 20])
# plant3 = cv2.bitwise_and(img[mask3rect[1] - 20:mask3rect[1] + mask3rect[3] + 20, mask3rect[0] - 20:mask3rect[0] + mask3rect[2] + 20], mask3[mask3rect[1] - 20:mask3rect[1] + mask3rect[3] + 20, mask3rect[0] - 20:mask3rect[0] + mask3rect[2] + 20])
plant1 = cv2.bitwise_and(img,mask1)
plant2 = cv2.bitwise_and(img,mask2)
plant3 = cv2.bitwise_and(img,mask3)
plant1 = ( (plant1 - numpy.min(plant1)) / (numpy.max(plant1) - numpy.min(plant1)) )
plant2 = ( (plant2 - numpy.min(plant2)) / (numpy.max(plant2) - numpy.min(plant2)) )
plant3 = ( (plant3 - numpy.min(plant3)) / (numpy.max(plant3) - numpy.min(plant3)) )
strElement = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,3))
leaves1 = numpy.copy(plant1)
leaves2 = numpy.copy(plant2)
leaves3 = numpy.copy(plant3)
cv2.erode(plant1, strElement, leaves1)
cv2.erode(plant2, strElement, leaves2)
cv2.erode(plant3, strElement, leaves3)
cv2.dilate(leaves1, strElement, leaves1)
cv2.dilate(leaves2, strElement, leaves2)
cv2.dilate(leaves3, strElement, leaves3)
roots1 = plant1 - leaves1
roots2 = plant2 - leaves2
roots3 = plant3 - leaves3
roots1 = ( (roots1 - numpy.min(roots1)) / (numpy.max(roots1) - numpy.min(roots1)) )
roots2 = ( (roots2 - numpy.min(roots2)) / (numpy.max(roots2) - numpy.min(roots2)) )
roots3 = ( (roots3 - numpy.min(roots3)) / (numpy.max(roots3) - numpy.min(roots3)) )

cv2.imshow('despues', roots1)
cv2.waitKey()
#Root 1
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask1[i,j] != 0):
            sum = sum + roots1[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots1, 3*avg, 1, cv2.THRESH_TOZERO, roots1)
roots1 = ricci(roots1, [3,15])
cv2.imshow('antes', roots1)
cv2.waitKey()
kernel = numpy.ones((3,3))/9
cv2.filter2D(roots1, -1, kernel, roots1)
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask1[i,j] != 0):
            sum = sum + roots1[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots1, 4*avg, 1, cv2.THRESH_BINARY, roots1)
print('Ricci 1 terminado')

#Root2
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask2[i,j] != 0):
            sum = sum + roots2[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots2, 3*avg, 1, cv2.THRESH_TOZERO, roots2)
roots2 = ricci(roots2, [3,15])
kernel = numpy.ones((3,3))/9
cv2.filter2D(roots2, -1, kernel, roots2)
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask2[i,j] != 0):
            sum = sum + roots2[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots2, 4*avg, 1, cv2.THRESH_BINARY, roots2)
print('Ricci 2 terminado')

#Root 3
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask3[i,j] != 0):
            sum = sum + roots3[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots3, 3*avg, 1, cv2.THRESH_TOZERO, roots3)
roots3 = ricci(roots3, [3,15])
kernel = numpy.ones((3,3))/9
cv2.filter2D(roots3, -1, kernel, roots3)
count = 0
sum = 0
for i in range(width):
    for j in range(height):
        if (mask3[i,j] != 0):
            sum = sum + roots3[i,j]
            count = count + 1
avg = sum/count
cv2.threshold(roots3, 4*avg, 1, cv2.THRESH_BINARY, roots3)
print('Ricci 3 terminado')

roots1[: , mask1rect[0]-5:mask1rect[0] + 20] = 0;
roots1[: , mask1rect[0] + mask1rect[2]-15:mask1rect[0] + mask1rect[2]+5] = 0;
roots2[: , mask2rect[0]-5:mask2rect[0] + 12] = 0;
roots2[: , mask2rect[0] + mask2rect[2]-10:mask2rect[0] + mask2rect[2]+5] = 0;
roots3[: , mask3rect[0]-5:mask3rect[0] + 12] = 0;
roots3[: , mask3rect[0] + mask3rect[2]-10:mask3rect[0] + mask3rect[2]+5] = 0;

newRoots1 = numpy.zeros(roots1.shape)
newRoots1 = eraseObjects(numpy.uint8(roots1*255), 20)
newRoots1 = eraseObjects(numpy.uint8(newRoots1), 10)
cv2.threshold(newRoots1, 250, 255, cv2.THRESH_BINARY, newRoots1)
print ("Objetos borrados en roots1")

newRoots2 = numpy.zeros(roots2.shape)
newRoots2 = eraseObjects(numpy.uint8(roots2*255), 20)
newRoots2 = eraseObjects(numpy.uint8(newRoots2), 10)
cv2.threshold(newRoots2, 250, 255, cv2.THRESH_BINARY, newRoots2)
print ("Objetos borrados en roots2")

newRoots3 = numpy.zeros(roots3.shape)
newRoots3 = eraseObjects(numpy.uint8(roots3*255), 20)
newRoots3 = eraseObjects(numpy.uint8(newRoots3), 10)
cv2.threshold(newRoots3, 250, 255, cv2.THRESH_BINARY, newRoots3)
print ("Objetos borrados en roots3")

print('Roots creadas y gris normalizado')
print('Generando Imagenes...')
riccifile1 = os.path.join(root, 'ricci/' + mask + '_ricci1.png')
riccifile2 = os.path.join(root, 'ricci/' + mask + '_ricci2.png')
riccifile3 = os.path.join(root, 'ricci/' + mask + '_ricci3.png')
cv2.imwrite(riccifile1, newRoots1, (cv2.IMWRITE_PNG_COMPRESSION, 0))
cv2.imwrite(riccifile2, newRoots2, (cv2.IMWRITE_PNG_COMPRESSION, 0))
cv2.imwrite(riccifile3, newRoots3, (cv2.IMWRITE_PNG_COMPRESSION, 0))
cv2.waitKey()
cv2.destroyAllWindows()
