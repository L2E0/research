import os
import cv2
import numpy as np 


path = "data-beach_resized"
w = np.array((240,240,240))

for file in os.listdir(path):
    if file != ".DS_Store":
        filepath = path + "/" + file
        src = cv2.imread(filepath, 1)
        if all(w < src[0, 0]) & all(w < src[255, 0]) & all(w < src[0, 255]) & all(w < src[255, 255]):
            os.remove(filepath)
