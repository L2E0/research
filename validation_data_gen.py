import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator as Gen

def Load_valid(category, size, hs, cov=None):
    path = "valid_" + category
    mono_list = []
    h_list = []
    s_list = []


    while True:
        for file in os.listdir(path):
            if file != ".DS_Store":
                filepath = path + "/" + file
                src = cv2.imread(filepath, 1)
                src = cv2.resize(src, (size, size))
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                h_list.append(np.ravel(hsv[:,:,0] / 180.0))
                s_list.append(np.ravel(hsv[:,:,1] / 255.0))
                gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if cov == True:
                    gry_cov = np.cov(gry) / np.max(np.absolute(np.cov(gry)))
                    mono_list.append(np.ravel(np.append(gry/255.0, gry_cov)))
                else:
                    mono_list.append(np.ravel(gry))
                if len(h_list) == 5:
                    h_list = np.array(h_list)
                    s_list = np.array(s_list)
                    mono_list = np.array(mono_list)
                    yield (mono_list, [h_list, s_list])
                    mono_list = []
                    h_list = []
                    s_list = []
