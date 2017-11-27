from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import os
from keras import optimizers as op
from keras.optimizers import Adam
import numpy as np
from datetime import *
from keras.utils import plot_model
from skimage.measure import compare_ssim as ssim
import cv2
import model
import predict
from data_gen import *


class ColorizationModel:
    def __init__(self):
        self.generator = model.Generator()
        self.discriminator = model.Discriminator()
        self.d_on_g = model.Generator_containing_discriminator(self.generator, self.discriminator)
        d_optim = Adam(lr=1e-5, beta_1=0.1)
        g_optim = Adam(lr=2e-4, beta_1=0.5)
        self.discriminator.trainable = False
        self.d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def load_weights(self):
        self.generator.load_weights('generator')
        #self.discriminator.load_weights('discriminator')

    def plot(self):
        plot_model(self.generator, ("generator.png"), show_shapes=True)
        plot_model(self.discriminator, ("discriminator.png"), show_shapes=True)

    def train(self, category, gen, val_gen, batch_size=32, step_size=100, epochs=100, offset=0):
        gen = batchgen(gen, batch_size)
        path = "valid_" + category
        val_size = 100
        val_gen = chunk(val_gen, val_size)

        pred_path = 'test_%s' % (category)
        pred_gen = xygen(pred_path, img2bgr)
        pred_gen = batchgen(pred_gen, 1)


        for epoch, steps in enumerate(epochgen(gen, epochs, step_size)):
            pred_gen = ((self.generator.predict(np.array([x]), verbose=0)[0], y) for x, y in next(val_gen))
            mapper = lambda pair:ssim(*pair, multichannel=True)
            evaluation = sum(map(mapper, pred_gen))
            evaluation /= val_size
            print("evaluation: ", evaluation)
            print("epoch: ", epoch+offset+1)
            for step, (x, y) in enumerate(steps):
                generated_image = self.generator.predict(x, verbose=0)
                X = np.concatenate((y, generated_image))
                y = [1] * batch_size + [0] * batch_size
                d_loss = self.discriminator.train_on_batch(X, y)
                print("step %d d_loss : %f" % (step+1, d_loss))
                label = [1] * batch_size
                g_loss = self.d_on_g.train_on_batch(x, label)
                print("step %d g_loss : %f" % (step+1, g_loss))

            f = open('gan.txt', 'w')
            f.write('%d\n' % (epoch+offset+1))
            f.write('d_loss%f\n' % (d_loss))
            f.write('g_loss%f\n' % (g_loss))
            f.close()
            self.generator.save_weights('generator', True)
            self.discriminator.save_weights('discriminator', True)

            #predict.Predict_BGR(self.generator, pred_gen, category, 'predictions/SRGAN', count_file(pred_path))

    def pre_train(self, gen, val_gen):
        gen = batchgen(gen, 8)
        val_gen = batchgen(val_gen, 8)
        self.generator.compile(loss='mean_squared_error', optimizer=Adam())
        checkpointer = ModelCheckpoint(filepath='generator', verbose=1, save_best_only=True)
        self.generator.fit_generator(gen, 100, epochs=30, verbose=1
                ,validation_data=val_gen, validation_steps=10)
        self.generator.save_weights('generator', True)

    def predict(self, category, epoch, transformer):
        path = 'test_%s_orig' % (category)
        gen = xygen(path, transformer)
        gen = batchgen(gen, 1)
        pre = 'predictions/SRGAN'
        #os.mkdir(pre)
        plot_model(self.generator, ("%s/generator.png" % (pre)), show_shapes=True)
        plot_model(self.discriminator, ("%s/discriminator.png" % (pre)), show_shapes=True)
        predict.Predict_BGR(self.generator, gen, category, pre, count_file(path))

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

