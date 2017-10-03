import os
import numpy as np
import cv2
def Load_bgr(category):
    mono_list = []
    red_list = []
    blue_list = []
    green_list = []
    path = "train_" + category

    for file in os.listdir(path):
        if file != ".DS_Store":
            filepath = path + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.resize(src, (256, 256))
            #bgr = np.array(cv2.split(src))
            blue_list.append(np.ravel(src[:,:,0] / 255.0))
            green_list.append(np.ravel(src[:,:,1] / 255.0))
            red_list.append(np.ravel(src[:,:,2] / 255.0))
            gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            mono_list.append(np.ravel(gry / 255.0))

    blue_list = np.array(blue_list)
    green_list = np.array(green_list)
    red_list = np.array(red_list)
    mono_list = np.array(mono_list)


    return blue_list, green_list, red_list, mono_list

def Load_hsv(category):
    mono_list = []
    h_list = []
    s_list = []
    v_list = []#mono
    path = "train_" + category

    for file in os.listdir(path):
        if file != ".DS_Store":
            filepath = path + "/" + file
            src = cv2.imread(filepath, 1)
            #src = cv2.resize(src, (256, 256, 3))
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            #hsv = np.array(cv2.split(hsv))
            h_list.append(np.ravel(hsv[:,:,0] / 180.0))
            s_list.append(np.ravel(hsv[:,:,1] / 255.0))
            v_list.append(np.ravel(hsv[:,:,2] / 255.0))
            gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            mono_list.append(np.ravel(gry / 255.0))

    h_list = np.array(h_list)
    s_list = np.array(s_list)
    v_list = np.array(v_list)
    mono_list = np.array(mono_list)


    return h_list, s_list, mono_list

def Load_cov(category):
    mono_list = []
    h_list = []
    s_list = []
    path = "train_" + category

    for file in os.listdir(path):
        if file != ".DS_Store":
            filepath = path + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.resize(src,(128, 128))
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            #h = hsv[:,:,0]
            #s = hsv[:,:,1]
            h_list.append(np.ravel(hsv[:,:,0] / 180.0))
            s_list.append(np.ravel(hsv[:,:,1] / 255.0))

            #h_cov = np.cov(h) / np.max(np.absolute(np.cov(h)))
            #s_cov = np.cov(s) / np.max(np.absolute(np.cov(s)))
            #h_list.append(np.ravel(np.append(h/180.0, h_cov)))
            #s_list.append(np.ravel(np.append(s/255.0, s_cov)))

            gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gry_cov = np.cov(gry) / np.max(np.absolute(np.cov(gry)))
            mono_list.append(np.ravel(np.append(gry/255.0, gry_cov)))

    h_list = np.array(h_list)
    s_list = np.array(s_list)
    mono_list = np.array(mono_list)



    return h_list, s_list, mono_list
