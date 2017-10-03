import os
import cv2


path = "data-beach"
dst = "data-beach_resized"
os.mkdir(dst)

for file in os.listdir(path):
    if file != ".DS_Store":
        filepath = path + "/" + file
        src = cv2.imread(filepath, 1)
        src = cv2.resize(src, (256, 256))
        filepath = dst + "/" + file
        cv2.imwrite(filepath, src)