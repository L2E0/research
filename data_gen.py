import itertools
from keras.preprocessing.image import ImageDataGenerator as Gen
import numpy as np
import os
import cv2

def chunk(gen, size):
    while True:
        yield itertools.islice(gen, size)

def xygen(path, **args):
    datagen = Gen(args)
    imggen = datagen.flow_from_directory(directory='./', classes=[path], batch_size=1, class_mode=None, shuffle=False)
    for src in imggen:
        src = src[0]
        src = np.flip(src, axis=2)
        gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = np.array(src / 255.0)
        gry = np.array(gry / 255.0)
        gry = np.expand_dims(gry, axis=2)
        yield gry, src

def batchgen(gen, batch_size):
    for xylist in chunk(gen, batch_size):
        x, y = zip(*xylist)
        x = np.array(list(x))
        y = np.array(list(y))
        yield x, y

def epochgen(gen, epochs, step_size):
    stepgen = chunk(gen, step_size)
    for step_size in itertools.islice(stepgen, epochs):
        yield step_size

def count_file(path):
    return len(os.listdir(path))

