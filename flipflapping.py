import cv2
import os

for file in os.listdir("train_beach"):
    if file != ".DS_Store":
        filepath = "train_beach/" + file
        src=cv2.imread(filepath, 1)
        src_u = cv2.flip(src, 0)
        src_f = cv2.flip(src, 1)
        src_uf = cv2.flip(src, -1)
        filename, ext = os.path.splitext(filepath)
        filename = filename + ".png"

        cv2.imwrite(filename.replace('.', "_u."), src_u, [16,0])
        cv2.imwrite(filename.replace('.', "_f."), src_f, [16,0])
        cv2.imwrite(filename.replace('.', "_uf."), src_uf, [16,0])