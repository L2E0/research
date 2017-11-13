import os
import cv2
import numpy as np 


path = "train_beach"
w = np.array((230,230,230))

for file in os.listdir(path):
    if file != ".DS_Store":
        filepath = path + "/" + file
        src = cv2.imread(filepath, 1)
        if all(w < src[0, 0]) and all(w < src[255, 0]) and all(w < src[0, 255]) and all(w < src[255, 255]):
            os.remove(filepath)
