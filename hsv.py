# -*- coding: utf-8 -*-
import cv2
import numpy as np
src = cv2.imread('144723456_cfe70f598a.jpg', 1)

gry = np.zeros(src.shape, dtype = np.uint8)
gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)


for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        src[i, j][1] = 0

src = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imshow('result', gry)
cv2.imwrite('s0.png', src)
print np.ravel(gry)
print src
cv2.waitKey(0)
