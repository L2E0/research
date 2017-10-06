import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator as Gen

def Load_valid_h(category):
    path = "valid_" + category
    mono_list = []
    h_list = []


    while True:
        for file in os.listdir(path):
            if file != ".DS_Store":
                filepath = path + "/" + file
                src = cv2.imread(filepath, 1)
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                h_list.append(np.ravel(hsv[:,:,0] / 180.0))
                gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                mono_list.append(np.ravel(gry / 255.0))
                if len(h_list) == 100:
                    h_list = np.array(h_list)
                    mono_list = np.array(mono_list)
                    yield (mono_list, h_list)
                    mono_list = []
                    h_list = []

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




