from PIL import Image
import os
from os.path import join
import numpy as np

path = "valid_beach"
for file in os.listdir(path):
    file = join(path, file)
    im = Image.open(file)
    im = np.asarray(im, dtype='float32')
    print("%s, %s" % (file, type(im)))
