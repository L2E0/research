from keras.models import Sequential, Model
from keras.layers import advanced_activations
from keras.layers import Input, Dense, Lambda
from keras.layers import noise, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
#from skimage.measure import compare_ssim as ssim
from keras.layers.core import Dropout, Activation, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers as op
from keras import backend as K
from keras import losses
import tensorflow as tf
import numpy as np

act = 'relu'

def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses

tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video

Super-Resolution Using an Efficient Sub-Pixel

Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def advanced_mse(y_true, y_pred):
    y_true = tf.minimum(abs(y_pred-y_true), 1-(abs(y_pred-y_true))) + y_pred
    return losses.mean_squared_error(y_true, y_pred)

def mlp():
    mono_input = Input(shape=(65536, ), name='input')
    x = Dense(2048, activation=act)(mono_input)
    x = Dense(2048, activation=act)(x)
    h = Dense(1024, activation=act)(x)
    h = Dense(1024, activation=act)(x)
    h_output = Dense(65536, activation=act, name='h')(h)
    s = Dense(1024, activation=act)(x)
    s = Dense(1024, activation=act)(x)
    s_output = Dense(65536, activation=act, name='s')(s)
    model = Model(inputs=mono_input, outputs=[h_output, s_output])
    model.compile(loss=advanced_mse, optimizer=op.Adam())

    return model


def Build():
    mono_input = Input(shape=(65536, ), name='input')
    x = Conv2D(48, 2, padding='same', activation=act)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(48, 3, padding='same', activation=act)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x =Conv2D(96, 2, padding='same', activation=act)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(96, 3, padding='same', activation=act)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(192, 2, padding='same', activation=act)(x)
    h = Dense(2048, activation=act)(x)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    h = Dropout(0.5)(h)
    s = Dense(2048, activation=act)(x)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    s = Dropout(0.5)(s)
    h_output = Dense(65536, activation='sigmoid', name='h')(h)
    s_output = Dense(65536, activation='sigmoid', name='s')(s)

    model = Model(inputs=mono_input, outputs=[h_output, s_output])
    model.compile(loss='mse', optimizer=op.Adam())

    return model


def Build_128():


    inputs = Input(shape=(32768, ))
    x = Dense(4096, kernel_initializer='zeros', bias_initializer='zeros', activation=act)(inputs)
    x = Dense(4096, kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    predictions = Dense(16384, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='mse', optimizer=op.Adam(), metrics=['mae'])

    return model

def combine():
    h_input = Input(shape=(16384, ), name='h_input')
    h = Dense(4096, activation=act)(h_input)
    h_output = Dense(4096, activation=act, name='h_output')(h)

    s_input = Input(shape=(16384, ), name='s_input')
    s = Dense(4096, activation=act)(s_input)
    s_output = Dense(4096, activation=act, name='s_output')(s)

    x = concatenate([h_output, s_output])

    x = Dense(4096, activation=act)(x)
    main_output = Dense(4096, activation=act, name='main_output')(x)

    model = Model(inputs=[h_input, s_input], outputs=main_output)
    model.compile(loss='mse', optimizer=op.Adam(), metrics=['mae'])

    return model

def Generator():
    #inputs = Input(shape=(256, 256, 1), name='input')
    #x = Reshape((65536,))(inputs)
    #x = BatchNormalization()(x)
    #x = Activation(act)(x)
    #x = Dense(1024, activation=act)(x)
    #x = BatchNormalization()(x)
    #x = Dense(1024, activation=act)(x)
    #x = BatchNormalization()(x)
    #x = Dense(64*64*36, activation=act)(x)
    #x = BatchNormalization()(x)
    #x = Reshape((64, 64, 36))(x)
    #x = UpSampling2D((2, 2))(x)
    #x = Conv2D(18, (5, 5), border_mode='same')(x)
    #x = BatchNormalization()(x)
    #x = Activation(act)(x)
    #x = UpSampling2D((2, 2))(x)
    #predictions = Conv2D(3, (5, 5), border_mode='same', activation='sigmoid')(x)
    inputs = Input(shape=(256, 256, 1), name='input')
    x = Conv2D(3, 5, strides=2, activation=act, padding='same')(inputs)
    x = Conv2D(3, 5, strides=2, activation=act, padding='same')(x)
    #x = Conv2D(512, 5, strides=2, activation=act, padding='same')(x)
    #x = Conv2D(1024, 5, strides=2, activation=act, padding='same')(x)
    #x = Conv2D(1024, 5, strides=2, activation=act, padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation=act)(x)
    x = Dense(4096, activation=act)(x)
    x = Dense(4096, activation=act)(x)
    x = Dense(4096, activation=act)(x)
    x = Dropout(0.5)(x)
    x = Dense(32768, activation=act)(x)
    x = Reshape((8, 8, 512))(x)
    x = SubpixelConv2D((8, 8, 512), scale=2)(x)
    x = Conv2D(256, 5, strides=1, activation=act, padding='same')(x)
    x = SubpixelConv2D((16, 16, 256), scale=2)(x)
    x = Conv2D(128, 5, strides=1, activation=act, padding='same')(x)
    x = SubpixelConv2D((32, 32, 128), scale=2)(x)
    x = Conv2D(64, 5, strides=1, activation=act, padding='same')(x)
    x = SubpixelConv2D((64, 64, 64), scale=2)(x)
    x = Conv2D(32, 5, strides=1, activation=act, padding='same')(x)
    x = SubpixelConv2D((128, 128, 32), scale=2)(x)
    predictions = Conv2D(3, 5, strides=1, activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def Discriminator():
    lrelu = LeakyReLU(0.3)
    inputs = Input(shape=(256, 256, 3), name='input')
    x = Conv2D(36, (5, 5),
                subsample=(2, 2),
                border_mode='same',
                activation=lrelu)(inputs)
    x = Conv2D(72, (5, 5), subsample=(2, 2), activation=lrelu)(x)
    x = Conv2D(108, (5, 5), subsample=(2, 2), activation=lrelu)(x)
    x = Conv2D(108, (5, 5), subsample=(2, 2), activation=lrelu)(x)
    x = Conv2D(108, (5, 5), subsample=(2, 2), activation=lrelu)(x)
    x = Flatten()(x)
    x = Dense(4096, activation=lrelu)(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model

def Generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
