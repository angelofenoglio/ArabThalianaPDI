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

def detectLines(image, windowRange):
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
