import cv2
import numpy as np
import sys


def eraseObjects(img, npixel):
    objects = np.zeros(img.shape, np.uint8)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if (len(contours[i]) < npixel):
            for j in range(len(contours[i])):
                x = contours[i][j][0][0]
                y = contours[i][j][0][1]
                cv2.circle(objects, (x, y), 1, 255, -1)
                # print(copyImg[y, x])
                # objects[y, x] = 255
    cv2.imshow("Objetos removibles", objects)
    return img - objects


filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
copyImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

cv2.imshow("Original",img)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
EE2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2,2))
EE3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,3))
done = False

copyImg = eraseObjects(copyImg, 20)
copyImg = eraseObjects(copyImg, 10)
copyImg = eraseObjects(copyImg, 5)

copyImg = cv2.morphologyEx(copyImg,cv2.MORPH_CLOSE,EE3)
cv2.imshow("Objetos removidos",copyImg)

done = False
while( not done):
    eroded = cv2.erode(copyImg,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(copyImg,temp)
    skel = cv2.bitwise_or(skel,temp)
    copyImg = eroded.copy()

    zeros = size - cv2.countNonZero(copyImg)
    if zeros==size:
        done = True

skel2 = cv2.morphologyEx(skel,cv2.MORPH_CLOSE,EE2)
cv2.imshow("Skeleton Sin Objetos",skel)

cv2.waitKey(0)
cv2.destroyAllWindows()