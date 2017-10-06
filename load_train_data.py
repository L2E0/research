import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator as Gen
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

def Load_gen(category, batch_size, hs):
    path = "train_" + category
    mono_list = []
    y_list = []


    datagen = Gen(horizontal_flip=True,
     vertical_flip=True,
     rotation_range=180,
     width_shift_range=0.2,
     height_shift_range=0.2,
     zoom_range=0.3)
    imggen = datagen.flow_from_directory(directory='./', classes=[path], batch_size=1, class_mode=None)

    for batch in imggen:
        img = np.roll(batch[0], -1, axis=2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if hs == 'h':
            y_list.append(np.ravel(hsv[:,:,0] / 180.0))
        elif hs == 's':
            y_list.append(np.ravel(hsv[:,:,1] / 255.0))
        mono_list.append(np.ravel(gry / 255.0))
        if len(y_list) == batch_size:
            y_list = np.array(y_list)
            mono_list = np.array(mono_list)
            yield (mono_list, y_list)
            mono_list = []
            y_list = []



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
