#!/usr/bin/env python3

import cv2
import sys
import numpy
from collections import deque
import math


def neighbours(x,y,image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def notSoNeighbours(x,y,image):
    img = image
    return [ img[x][y+2], img[x+1][y+2], img[x+2][y+2], img[x+2][y+1], img[x+2][y], img[x+2][y-1], img[x+2][y-2], img[x+1][y-2], img[x][y-2],
             img[x-1][y-2], img[x-2][y-2], img[x-2][y-1], img[x-2][y], img[x-2][y+1], img[x-2][y+2], img[x-1][y+2] ]

def neighboursCoors(x,y):
    y, x = x, y
    return [ [y-1, x], [y-1, x+1], [y, x+1], [y+1, x+1], [y+1, x], [y+1, x-1], [y, x-1], [y-1, x-1] ]

def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )


def getEndPoints(skeleton):
    height, width = skeleton.shape
    endPoints = list()
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if (skeleton[i,j] and numpy.sum(neighbours(i,j,skeleton)) == 1):
                endPoints.append([i,j])
    return endPoints


def intersectionPoints(nLabels, labels):
    height, width = labels.shape
    intPoints = list()
    # aca si el cluster tiene 3 o 4 pixeles sacar el centroide, deberia andar casi siempre
    for label in range(1, nLabels):
        cluster = numpy.zeros_like(labels)
        cluster[numpy.where(labels == label)] = 1
        clusterNeighbours = list()
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if (cluster[i, j]):
                    clusterNeighbours.append([i, j, numpy.sum(neighbours(i, j, cluster))])
        maxindx = numpy.argmax(clusterNeighbours, axis=0)
        maxindx = maxindx[-1]
        intPoints.append(clusterNeighbours[maxindx][0:2])
    return intPoints


def getHighestEndPoint(endPoints):
    indx = numpy.argmin(endPoints,axis=0)
    indx = indx[0]
    return endPoints[indx]


def intersectionIndex(point, pointsPerLabel):
    labels = len(pointsPerLabel)
    index = -1
    for i in range(labels):
        found = point in pointsPerLabel[i]
        if found:
            index = i
            break
    return index


def nextNode(point, visited, label, endPoints, interPoints, nodes, connections, lastNode):
    while True:
        if point in endPoints:
            nodes.append(point)
            connections.append([nodes.index(lastNode), len(nodes) - 1])
            print('Punto final')
            return
        else:
            n = neighboursCoors(point[0], point[1])
            if point in interPoints:
                flag = point not in nodes
                if flag:
                    nodes.append(point)
                connections.append([nodes.index(lastNode), len(nodes) - 1])
                print('Interseccion')
                if flag:
                    nextPoints = list()
                    for i in range(0,8,2):
                        if label[n[i][0], n[i][1]] and not n[i] in visited:
                            nextPoints.append(n[i])
                            visited.append(n[i])
                    for i in range(len(nextPoints)):
                        print('Recursiva')
                        visited.append(point)
                        nextNode(nextPoints[i], visited, label, endPoints, interPoints, nodes, connections, point)
                    nextPoints = list()
                    for i in range(1,8,2):
                        if label[n[i][0], n[i][1]] and not n[i] in visited:
                            nextPoints.append(n[i])
                            visited.append(n[i])
                    for i in range(len(nextPoints)):
                        print('Recursiva')
                        visited.append(point)
                        nextNode(nextPoints[i], visited, label, endPoints, interPoints, nodes, connections, point)
                    return
            else:
                # en el label 1 con el loop al tratar de crear el ultimo edge se tiene que pasar por un pixel visitado, entonces no ecuentra nada y falla en la siguiente vuelta
                visited.append(point)
                nextPoint = list()
                if label[n[0][0], n[0][1]] and not n[0] in visited: nextPoint = n[0]
                if label[n[2][0], n[2][1]] and not n[2] in visited: nextPoint = n[2]
                if label[n[4][0], n[4][1]] and not n[4] in visited: nextPoint = n[4]
                if label[n[6][0], n[6][1]] and not n[6] in visited: nextPoint = n[6]
                if len(nextPoint) == 0:
                    if label[n[1][0], n[1][1]] and not n[1] in visited: nextPoint = n[1]
                    if label[n[3][0], n[3][1]] and not n[3] in visited: nextPoint = n[3]
                    if label[n[5][0], n[5][1]] and not n[5] in visited: nextPoint = n[5]
                    if label[n[7][0], n[7][1]] and not n[7] in visited: nextPoint = n[7]
                point = nextPoint

filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# img[408:415, 250:253] = 0
# img[316:318, 271:273] = 0
height, width = img.shape
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(numpy.uint8(img))
largestLabels = numpy.argpartition(stats[1:-1,-1], -3)[-3:] + 1
label1 = numpy.zeros_like(labels)
label2 = numpy.zeros_like(labels)
label3 = numpy.zeros_like(labels)
label1[numpy.where(labels == largestLabels[0])] = 1
label1 = numpy.uint8(label1)
label2[numpy.where(labels == largestLabels[1])] = 1
label2 = numpy.uint8(label2)
label3[numpy.where(labels == largestLabels[2])] = 1
label3 = numpy.uint8(label3)

roots = label1

endPoints = getEndPoints(roots)
intsctl = list()
for i in range(1, width - 1):
    for j in range(1, height - 1):
        if roots[i, j] and transitions(notSoNeighbours(i, j, roots)) > 2 and numpy.sum(neighbours(i, j, roots)) > 2:
            intsctl.append((i,j))
intersect = numpy.zeros_like(labels)
for index in range(len(intsctl)):
    intersect[intsctl[index][0], intsctl[index][1]] = 1
nintlabel, intlabels, intstats, centroids = cv2.connectedComponentsWithStats(numpy.uint8(intersect))

interPoints = intersectionPoints(nintlabel, intlabels)
#
# pointsPerLabel = list()
# for label in range(1,nintlabel):
#     pointsPerLabel.append(numpy.where(intlabels == label))
# initPoint = getHighestEndPoint(endPoints)
# endPoints.remove(initPoint)
# visited = deque()
# nodes = list()
# connections = list()
# nodes.append(initPoint)
# nextNode(initPoint, visited, roots, endPoints, interPoints, nodes, connections, initPoint)
# print(nodes)
# print(connections)

blank = 255*numpy.ones_like(roots)
label_hue = numpy.uint8(179*(roots + intersect*5))
for index in range(len(interPoints)):
    label_hue[interPoints[index][0], interPoints[index][1]] = 120
for index in range(len(endPoints)):
    label_hue[endPoints[index][0], endPoints[index][1]] = 50
withintersect = cv2.merge([label_hue, blank, blank])
withintersect = cv2.cvtColor(withintersect, cv2.COLOR_HSV2BGR)
withintersect[roots==0] = 0
cv2.imshow('intersect',withintersect)

# label_hue = numpy.uint8(179*labels/numpy.max(labels))
# blank_ch = 255*numpy.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
# # cvt to BGR for display
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
# # set bg label to black
# labeled_img[label_hue==0] = 0
#
# cv2.imshow('labeled.png', labeled_img)

cv2.waitKey()
