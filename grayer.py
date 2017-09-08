import os
import cv2

for file in os.listdir("test"):
    if file != ".DS_Store":
        filepath = "test/" + file

        src = cv2.imread(filepath, 1)
        src = cv2.resize(src, (256, 256))
        gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filepath, gry)
