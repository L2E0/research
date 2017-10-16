from keras.callbacks import EarlyStopping
from datetime import datetime
import os
from keras import optimizers as op
from keras.optimizers import Adam
import numpy as np
from data_gen import xygen, batchgen, epochgen, count_file, chunk
from keras.utils import plot_model
from skimage.measure import compare_ssim as ssim
import cv2
import model
import predict


class ColorizationModel:
    def __init__(self):
        self.generator = model.Generator()
        self.discriminator = model.Discriminator()
        self.d_on_g = model.Generator_containing_discriminator(generator, discriminator)
        d_optim = Adam(lr=1e-6, beta_1=0.1)
        g_optim = Adam(lr=3e-4, beta_1=0.5)
        self.d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def load_weights(self):
        self.generator.load_weights('generator')
        self.discriminator.load_weights('discriminator')

    def plot(self):
        plot_model(self.generator, ("generator.png"), show_shapes=True)
        plot_model(self.discriminator, ("discriminator.png"), show_shapes=True)

    def train(self, gen, val_gen, batch_size=32, step_size=100, epochs=100, offset=0):
        gen = batchgen(gen, batch_size)

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
                y = [np.array([1, 0])] * batch_size + [np.array([0, 1])] * batch_size
                y = np.array(y)
                d_loss = self.discriminator.train_on_batch(X, y)
                #print("step %d d_loss : %f" % (step+1, d_loss))
                label = [np.array([1, 0])] * batch_size
                label = np.array(label)
                g_loss = d_on_g.train_on_batch(x, label)
                #print("step %d g_loss : %f" % (step+1, g_loss))
                if step % 50 == 49:
                    f = open('epoch.txt', 'w')
                    f.write('%d\n' % (epoch+offset+1))
                    f.write('d_loss%f\n' % (d_loss))
                    f.write('g_loss%f\n' % (g_loss))
                    f.close()
                    self.generator.save_weights('generator', True)
                    self.discriminator.save_weights('discriminator', True)

    def pre_train(self, gen):
        gen = batchgen(gen, batch_size)
        self.generator.fit_generator(gen, 100, epochs=100, verbose=1)

    def predict(self, gen, category, epoch):
        gen = batchgen(gen, batch_size)
        pre = 'predictions/epoch_%d' % (epoch)
        os.mkdir(pre)
        predict.Predict_BGR(self.generator, category, pre)

