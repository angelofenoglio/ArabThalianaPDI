#!/usr/bin/env python3

import cv2
import sys
import numpy
from os import system
import os.path
from collections import deque


def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]


def notSoNeighbours(x, y, image):
    img = image
    return [ img[x][y+2], img[x+1][y+2], img[x+2][y+2], img[x+2][y+1], img[x+2][y], img[x+2][y-1], img[x+2][y-2], img[x+1][y-2], img[x][y-2],
             img[x-1][y-2], img[x-2][y-2], img[x-2][y-1], img[x-2][y], img[x-2][y+1], img[x-2][y+2], img[x-1][y+2] ]


def neighboursCoors(x, y):
    y, x = x, y
    return [ [y-1, x], [y-1, x+1], [y, x+1], [y+1, x+1], [y+1, x], [y+1, x-1], [y, x-1], [y-1, x-1] ]


def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )


def getEndPoints(skeleton):
    height, width = skeleton.shape
    endPoints = list()
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if skeleton[i, j] and numpy.sum(neighbours(i, j, skeleton)) == 1:
                endPoints.append([i, j])
    return endPoints


def intersectionPoints(nLabels, labels, root):
    height, width = labels.shape
    intPoints = list()
    label = 1
    while label < nLabels:
        cluster = numpy.zeros_like(labels)
        cluster[numpy.where(labels == label)] = 1
        clusterSize = numpy.sum(cluster)
        points = list()
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if cluster[i, j]:
                    points.append([i, j])

        if clusterSize > 5 or len(numpy.unique([p[0] for p in points])) > 3 or len(numpy.unique([p[1] for p in points])) > 3:
            if len(numpy.unique([p[0] for p in points])) > len(numpy.unique([p[1] for p in points])):
                max = numpy.max([p[0] for p in points])
                min = numpy.min([p[0] for p in points])
                col = 0
            else:
                max = numpy.max([p[1] for p in points])
                min = numpy.min([p[1] for p in points])
                col = 1
            midPoint = (max+min)/2
            newLabelPoints = 0
            i = 0
            while i < len(points):
                if points[i][col] > midPoint:
                    labels[points[i][0], points[i][1]] = nLabels
                    points.remove(points[i])
                    newLabelPoints += 1
                    i -= 1
                i += 1
            nLabels += 1
            clusterSize -= newLabelPoints

        if clusterSize == 1:
            intPoints.append(points[0])

        if clusterSize == 2:
            if len(numpy.unique([p[0] for p in points])) == 2 and len(numpy.unique([p[1] for p in points])) == 2:
                if root[points[0][0], points[1][1]]:
                    intPoints.append([points[0][0], points[1][1]])
                elif root[points[1][0], points[0][1]]:
                    intPoints.append([points[1][0], points[0][1]])
                else:
                    clusterSize = 5
            else:
                clusterSize = 5 #si ninguno de los dos casos funciona que se decida por vecinos

        if 3 <= clusterSize <= 4:
            p = numpy.sum(points, axis=0)
            px = int(numpy.round(p[0]/len(points)))
            py = int(numpy.round(p[1]/len(points)))
            if [px, py] in points:
                intPoints.append([px, py])
            else:
                clusterSize = 5

        if clusterSize > 4:
            clusterNeighbours = list()
            for i in range(len(points)):
                clusterNeighbours.append(numpy.sum(neighbours(points[i][0], points[i][1], cluster)))
            maxindx = numpy.argmax(clusterNeighbours)
            intPoints.append(points[maxindx])

        label += 1 #endwhile

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


def getNearestIntersectionPoint(point, intersectionPoints):
    winDist = numpy.linalg.norm(numpy.array(intersectionPoints[0])-numpy.array(point[:]))
    index = 0
    for i in range(1, len(intersectionPoints)):
        dist = numpy.linalg.norm(numpy.array(intersectionPoints[i])-numpy.array(point))
        if dist < winDist:
            winDist = dist
            index = i
    return intersectionPoints[index]


def nextNode(point, visited, label, endPoints, interPoints, nodes, connections, lengths, lastNode):
    walked = 0
    while True:
        if point in endPoints:
            nodes.append(point)
            connections.append([nodes.index(lastNode), len(nodes) - 1])
            lengths.append(walked)
            print('Punto final')
            return
        else:
            if not point:
                nearestInterPoint = getNearestIntersectionPoint(visited[-1], interPoints)
                connections.append([nodes.index(lastNode), nodes.index(nearestInterPoint)])
                lengths.append(walked)
                return
            else:
                n = neighboursCoors(point[0], point[1])
                if point in interPoints:
                    flag = point not in nodes
                    if flag:
                        nodes.append(point)
                    connections.append([nodes.index(lastNode), len(nodes) - 1])
                    lengths.append(walked)
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
                            nextNode(nextPoints[i], visited, label, endPoints, interPoints, nodes, connections, lengths, point)
                        nextPoints = list()
                        for i in range(1,8,2):
                            if label[n[i][0], n[i][1]] and not n[i] in visited:
                                nextPoints.append(n[i])
                                visited.append(n[i])
                        for i in range(len(nextPoints)):
                            print('Recursiva')
                            visited.append(point)
                            nextNode(nextPoints[i], visited, label, endPoints, interPoints, nodes, connections, lengths, point)
                        return
                else:
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
                    walked += 1


def makeGraph(roots):
    width, height = roots.shape
    endPoints = getEndPoints(roots)
    intsctl = list()
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            if roots[i, j] and transitions(notSoNeighbours(i, j, roots)) > 2 and numpy.sum(neighbours(i, j, roots)) > 2:
                intsctl.append((i, j))
    intersect = numpy.zeros_like(labels)
    for index in range(len(intsctl)):
        intersect[intsctl[index][0], intsctl[index][1]] = 1
    nintlabel, intlabels, intstats, centroids = cv2.connectedComponentsWithStats(numpy.uint8(intersect))

    interPoints = intersectionPoints(nintlabel, intlabels, roots)

    pointsPerLabel = list()
    for label in range(1, nintlabel):
        pointsPerLabel.append(numpy.where(intlabels == label))
    initPoint = getHighestEndPoint(endPoints)
    endPoints.remove(initPoint)
    visited = deque()
    nodes = list()
    connections = list()
    lengths = list()
    nodes.append(initPoint)
    nextNode(initPoint, visited, roots, endPoints, interPoints, nodes, connections, lengths, initPoint)
    return nodes, connections, lengths


def drawGraph(canvas, nodes, connections):
    for edge in connections:
        p1 = tuple(reversed(nodes[edge[0]]))
        p2 = tuple(reversed(nodes[edge[1]]))
        cv2.line(canvas, p1, p2, (255, 255, 255), 1)
        cv2.circle(canvas, p1, 2, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 2, (0, 0, 255), -1)
    return canvas


filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
file = os.path.split(filename)
root = file[0]
root = os.path.dirname(root)
file = file[1]
file = os.path.splitext(file)
file = file[0]


nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(numpy.uint8(img))
largestLabels = numpy.argpartition(stats[1:, -1], -3)[-3:] + 1
label1 = numpy.zeros_like(labels)
label2 = numpy.zeros_like(labels)
label3 = numpy.zeros_like(labels)
label1[numpy.where(labels == largestLabels[0])] = 1
label1 = numpy.uint8(label1)
label2[numpy.where(labels == largestLabels[1])] = 1
label2 = numpy.uint8(label2)
label3[numpy.where(labels == largestLabels[2])] = 1
label3 = numpy.uint8(label3)

nodes1, connections1, lengths1 = makeGraph(label1)
nodes2, connections2, lengths2 = makeGraph(label2)
nodes3, connections3, lengths3 = makeGraph(label3)

width, height = img.shape
graph = numpy.zeros((width, height, 3), numpy.uint8)

graph = drawGraph(graph, nodes1, connections1)
graph = drawGraph(graph, nodes2, connections2)
graph = drawGraph(graph, nodes3, connections3)

graphpath = os.path.join('graphs/', file + '_graph.png')
graphpath = os.path.join(root, graphpath)

cv2.imwrite(graphpath, graph, (cv2.IMWRITE_PNG_COMPRESSION, 0))

# system('clear')
# print('Graph 1:')
# print('------------------------------------------')
# print('Nodes:')
# print(nodes1)
# print('Connections:')
# print(connections1)
# print('Lengths:')
# print(lengths1)
# print('')
# print('Graph 2:')
# print('------------------------------------------')
# print('Nodes:')
# print(nodes2)
# print('Connections:')
# print(connections2)
# print('Lengths:')
# print(lengths2)
# print('')
# print('Graph 3:')
# print('------------------------------------------')
# print('Nodes:')
# print(nodes3)
# print('Connections:')
# print(connections3)
# print('Lengths:')
# print(lengths3)
#
# cv2.imshow('Graphs', graph)
# cv2.waitKey()