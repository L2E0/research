import os
import numpy as np
import cv2
from prefetch_generator import background
import concurrent.futures
import itertools
import threading
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

@background(max_prefetch=320)
def Load_hsv(category, batch_size):
    path = "train_" + category
    mono_list = []
    h_list = []
    s_list = []


    datagen = Gen(horizontal_flip=True,
     vertical_flip=True,
     rotation_range=180,
     width_shift_range=0.2,
     height_shift_range=0.2,
     zoom_range=0.3)
    imggen = datagen.flow_from_directory(directory='./', classes=[path], batch_size=1, class_mode=None)

    for batch in imggen:
        img = np.flip(batch[0], axis=2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_list.append(np.ravel(hsv[:,:,0] / 360.0))
        s_list.append(np.ravel(hsv[:,:,1]))
        mono_list.append(np.ravel(gry / 255.0))
        if len(h_list) == batch_size:
            h_list = np.array(h_list)
            s_list = np.array(s_list)
            mono_list = np.array(mono_list)
            yield (mono_list, [h_list, s_list])
            mono_list = []
            h_list = []
            s_list = []


@background(max_prefetch=320)
def Load_cov(category, batch_size, hs):
    path = "train_" + category

    datagen = Gen(horizontal_flip=True,
     vertical_flip=True,
     rotation_range=180,
     width_shift_range=0.2,
     height_shift_range=0.2,
     zoom_range=0.3)
    imggen = datagen.flow_from_directory(directory='./', classes=[path], batch_size=1, class_mode=None)


    executor = concurrent.futures.ThreadPoolExecutor(32)
    while True:
        futures = [executor.submit(transform_img, img, hs) for img in itertools.islice(imggen,batch_size)]
        x_list = np.empty((batch_size, 32768))
        y_list = np.empty((batch_size, 16384))
        for i, future in enumerate(futures):
            x, y = future.result()
            x_list[i] = x
            y_list[i] = y
        yield (x_list, y_list)


def transform_img(img, hs):
    img = np.flip(img[0], axis=2)
    img = cv2.resize(img,(128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hs == 'h':
        y = np.ravel(hsv[:,:,0] / 360.0)
    elif hs == 's':
        y = np.ravel(hsv[:,:,1])
    gry_cov = np.cov(gry) / np.max(np.absolute(np.cov(gry)))
    x = np.ravel(np.append(gry/255.0, gry_cov))
    return x, y