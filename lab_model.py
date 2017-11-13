# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from datetime import datetime
import os
import argparse
from keras.optimizers import Adam
from keras.utils import plot_model


import load_train_data
import model
import plot
import validation_data_gen
from data_gen import *

class ColorizationModel:
    def __init__(self):
        self.mlp = model.mlp()

    def load_weights(self):
        self.mlp.load_weights('lab')

    def plot(self):
        plot_model(self.mlp, ("lab.png"), show_shapes=True)

    def train(self, category, gen, val_gen, batch_size=32, step_size=100, epochs=100, offset=0):
        train_gen = labgen(gen, batch_size)
        val_gen = labgen(val_gen, 10)
        history = self.mlp.fit_generator(
                train_gen, 
                steps_per_epoch = step_size,
                epochs = offset+epochs,
                validation_data = val_gen,
                validation_steps = 10,
                verbose=1,
                initial_epoch = offset,
        )
        self.mlp.save_weights('lab', True)
        f = open('epoch_lab.txt', 'w')
        f.write('%d\n' % (epochs+offset))
        f.close()

    def predict(self, category, offset):
        path = 'test_%s' % (category)
        gen = xygen(path, img2lab)
        pre = 'predictions/%s_lab_epoch_%d' % (category, offset)
        os.mkdir(pre)
        plot_model(self.mlp, ("%s/mlp.png" % (pre)), show_shapes=True)
        for i, img in zip(range(count_file(path)), gen):
            x, y = img
            x = np.ravel(x)
            a, b = self.mlp.predict_on_batch(np.array([x]))
            a = a[0]
            b = b[0]
            a = np.array(a*255.0, dtype='uint8')
            b = np.array(b*255.0, dtype='uint8')
            l = np.array(x*255.0, dtype='uint8')
            out = np.c_[l, a, b]
            out.resize((256, 256, 3))
            out = np.array(out, dtype='uint8')
            filepath = "%s/%d.png" % (pre, i)
            out = cv2.cvtColor(out, cv2.COLOR_Lab2BGR)
            cv2.imwrite(filepath, out)

            #h_ssim = ssim(h, h_org)
            #s_ssim = ssim(s, s_org)
            #pic_ssim = ssim(org, out, multichannel=True)

            filename = "%s/%d.png" % (pre, i)
            cv2.imwrite(filename, out)

    def summary(self):
        self.mlp.summary()

