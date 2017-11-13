import itertools
from keras.preprocessing.image import ImageDataGenerator as Gen
import numpy as np
import os
import cv2

def chunk(gen, size):
    while True:
        yield itertools.islice(gen, size)

def xygen(path, transformer, **args):
    datagen = Gen(args)
    imggen = datagen.flow_from_directory(directory='./', classes=[path], batch_size=1, class_mode=None, shuffle=False)
    for src in imggen:
        src = src[0]
        gry = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        src, gry = transformer(src, gry)
        gry = np.array(gry / 255.0)

        x = cv2.resize(src, (64, 64))
        yield x, src

def batchgen(gen, batch_size):
    for xylist in chunk(gen, batch_size):
        x, y = zip(*xylist)
        x = np.array(list(x))
        y = np.array(list(y))
        yield x, y

def hsvgen(gen, batch_size):
    for xylist in chunk(gen, batch_size):
        x, y = zip(*xylist)
        x = np.array(list(x))
        y = np.array(list(y))
        h = y[:,0]
        s = y[:,1]
        yield (x, [h, s])

def labgen(gen, batch_size):
    for xylist in chunk(gen, batch_size):
        x, y = zip(*xylist)
        x = np.array(list(x))
        y = np.array(list(y))
        a = y[:,0]
        b = y[:,1]
        yield (x, [a, b])

def epochgen(gen, epochs, step_size):
    stepgen = chunk(gen, step_size)
    for step_size in itertools.islice(stepgen, epochs):
        yield step_size

def count_file(path):
    return len(os.listdir(path))

def img2bgr(src, gry):
    src = np.flip(src, axis=2)
    return src / 255.0, np.expand_dims(gry, axis=2)
        

def img2hsv(src, gry):
    src = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    src = np.delete(src, 2, 2)
    src = np.transpose(src, (2, 0, 1))
    src = src.reshape(2, 65536)
    src = np.array(src)
    src[0] /= 360.0
    return src, np.ravel(gry) 

def img2lab(src, gry):
    src = cv2.cvtColor(src, cv2.COLOR_RGB2Lab)
    src = np.delete(src, 0, 2)
    src = np.transpose(src, (2, 0, 1))
    return src.reshape(2, 65536)/255.0, np.ravel(gry)
