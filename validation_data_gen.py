import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator as Gen

def Load_valid(category, size, hs, cov=None):
    path = "valid_" + category
    mono_list = []
    y_list = []


    while True:
        for file in os.listdir(path):
            if file != ".DS_Store":
                filepath = path + "/" + file
                src = cv2.imread(filepath, 1)
                src = cv2.resize(src, (size, size))
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                if hs == 'h':
                    y_list.append(np.ravel(hsv[:,:,0] / 360.0))
                elif hs == 's':
                    y_list.append(np.ravel(hsv[:,:,1] / 255.0))
                gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if cov == True:
                    gry_cov = np.cov(gry) / np.max(np.absolute(np.cov(gry)))
                    mono_list.append(np.ravel(np.append(gry/255.0, gry_cov)))
                else:
                    mono_list = gry
                if len(y_list) == 5:
                    y_list = np.array(y_list)
                    mono_list = np.array(mono_list)
                    yield (mono_list, y_list)
                    mono_list = []
                    y_list = []

def Load_valid_s(category):
    path = "valid_" + category
    mono_list = []
    s_list = []

    while True:
        for file in os.listdir(path):
            if file != ".DS_Store":
                filepath = path + "/" + file
                src = cv2.imread(filepath, 1)
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                s_list.append(np.ravel(hsv[:,:,1] / 255.0))
                gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                mono_list.append(np.ravel(gry / 255.0))
                if len(s_list) == 100:
                    s_list = np.array(s_list)
                    mono_list = np.array(mono_list)
                    yield (mono_list, s_list)
                    mono_list = []
                    s_list = []




