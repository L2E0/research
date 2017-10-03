import cv2
import os

for file in os.listdir("train_beach"):
    if file != ".DS_Store":
        filepath = "train_beach/" + file
        src=cv2.imread(filepath, 1)
        src_u = cv2.flip(src, 0)
        src_f = cv2.flip(src, 1)
        src_uf = cv2.flip(src, -1)

        #cv2.imwrite(filepath.replace('.', "_u."), src_u)
        cv2.imwrite(filepath.replace('.', "_f."), src_f)
        #cv2.imwrite(filepath.replace('.', "_uf."), src_uf)