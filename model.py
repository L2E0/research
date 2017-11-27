from keras.models import Sequential, Model
from keras.layers import advanced_activations
from keras.layers import Input, Dense, Lambda
from keras.layers import noise, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import add, multiply, average
#from skimage.measure import compare_ssim as ssim
from keras.layers.core import Dropout, Activation, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers as op
from keras import backend as K
from keras import losses
import tensorflow as tf
import numpy as np


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

def swish(x):
    return x * K.sigmoid(x)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def advanced_mse(y_true, y_pred):
    if tf.minimum(abs(y_pred-y_true), 1-(abs(y_pred-y_true))):
        y_true = tf.minimum(abs(y_pred-y_true), 1-(abs(y_pred-y_true))) + y_pred
    return losses.mean_squared_error(y_true, y_pred)

def advanced_relu(x):

    return K.relu((x))


def minus_one(x):
    return minus_one(x-1.0) if x > 1.0 else x

act = 'relu'

def mlp():
    mono_input = Input(shape=((65536,)), name='input')
    x1 = Dense(2048, activation=act)(mono_input)
    x = Dense(2048, activation=act)(x1)
    x = Dense(2048, activation=act)(x)
    x = Dense(2048, activation=act)(x)
    add1 = add([x1, x])
    x = Dense(2048, activation=act)(add1)
    x = Dense(2048, activation=act)(x)
    x = Dense(2048, activation=act)(x)
    add1 = add([add1, x])
    x = Dense(2048, activation=act)(add1)
    x = Dense(2048, activation=act)(x)
    x = Dense(2048, activation=act)(x)
    add1 = add([add1, x])
    x = add([add1, x1])
    x = Activation(act)(x)
    x = Dropout(0.5)(x)


    h1 = Dense(1024, activation=act)(x)
    h = Dense(1024, activation=act)(h1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([h1, h])
    h = Dense(1024, activation=act)(add1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([add1, h])
    h = Dense(1024, activation=act)(add1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([add1, h])
    h = Dense(1024, activation=act)(add1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([add1, h])
    h = Dense(1024, activation=act)(add1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([add1, h])
    h = Dense(1024, activation=act)(add1)
    h = Dense(1024, activation=act)(h)
    h = Dense(1024, activation=act)(h)
    add1 = add([add1, h])
    h = add([add1, h1])
    h = Activation(act)(h)

    h_output = Dense(65536, activation='sigmoid', name='h')(h)

    s1 = Dense(1024, activation=act)(x)
    s = Dense(1024, activation=act)(s1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([s1, s])
    s = Dense(1024, activation=act)(add1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([add1, s])
    s = Dense(1024, activation=act)(add1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([add1, s])
    s = Dense(1024, activation=act)(add1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([add1, s])
    s = Dense(1024, activation=act)(add1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([add1, s])
    s = Dense(1024, activation=act)(add1)
    s = Dense(1024, activation=act)(s)
    s = Dense(1024, activation=act)(s)
    add1 = add([add1, s])
    s = add([add1, s1])
    s = Activation(act)(s)

    s_output = Dense(65536, activation='sigmoid', name='s')(s)
    model = Model(inputs=mono_input, outputs=[h_output, s_output])
    model.compile(loss="mean_squared_error", optimizer=op.Adam())

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
    inputs = Input(shape=(64, 64, 3), name='input')
    x = Conv2D(64, 9, padding='same')(inputs)
    res1 = Activation(PReLU())(x)

    x = Conv2D(64, 3, padding='same')(res1)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    res = add([res1, x])

    x = Conv2D(64, 3, padding='same')(res)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    res = add([res, x])

    x = Conv2D(64, 3, padding='same')(res)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    res = add([res, x])

    x = Conv2D(64, 3, padding='same')(res)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    res = add([res, x])

    x = Conv2D(64, 3, padding='same')(res)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = add([res, x])

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, res1])

    x = Conv2D(256, 3, padding='same')(x)
    x = SubpixelConv2D((64, 64, 3), scale=2)(x)
    x = Activation(PReLU())(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = SubpixelConv2D((128, 128, 3), scale=2)(x)
    x = Activation(PReLU())(x)

    predictions = Conv2D(3, 9, padding='same')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def Generator2():
    inputs = Input(shape=(256, 256, 3), name='input')
    x = BatchNormalization()(inputs)
    x1 = Conv2D(16, 4, strides=2, padding='same', activation=act)(x)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(32, 4, strides=2, padding='same', activation=act)(x1)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(64, 4, strides=2, padding='same', activation=act)(x2)
    x3 = BatchNormalization()(x3)
    x4 = Conv2D(128, 4, strides=2, padding='same', activation=act)(x3)
    x4 = BatchNormalization()(x4)
    x5 = Conv2D(256, 4, strides=2, padding='same', activation=act)(x4)
    x5 = BatchNormalization()(x5)
    x6 = Conv2D(512, 4, strides=2, padding='same', activation=act)(x5)
    x6 = BatchNormalization()(x6)
    x7 = Conv2D(512, 4, strides=2, padding='same', activation=act)(x6)
    x7 = BatchNormalization()(x7)
    x8 = Conv2D(1024, 4, strides=2, padding='same', activation=act)(x7)
    x8 = BatchNormalization()(x8)
    x = Conv2DTranspose(512, 4, strides=2, padding='same', activation=act)(x8)
    x = BatchNormalization()(x)
    add1 = concatenate([x7, x])
    x = Conv2DTranspose(512, 4, strides=2, padding='same', activation=act)(add1)
    x = BatchNormalization()(x)
    add2 = concatenate([x6, x])
    x = Conv2DTranspose(256, 4, strides=2, padding='same', activation=act)(add2)
    x = BatchNormalization()(x)
    add3 = concatenate([x5, x])
    x = Conv2DTranspose(128, 4, strides=2, padding='same', activation=act)(add3)
    x = BatchNormalization()(x)
    add4 = concatenate([x4, x])
    x = Conv2DTranspose(64, 4, strides=2, padding='same', activation=act)(add4)
    x = BatchNormalization()(x)
    add5 = concatenate([x3, x])
    x = Conv2DTranspose(32, 4, strides=2, padding='same', activation=act)(add5)
    x = BatchNormalization()(x)
    add6 = concatenate([x2, x])
    x = Conv2DTranspose(16, 4, strides=2, padding='same', activation=act)(add6)
    x = BatchNormalization()(x)
    add7 = concatenate([x1, x])
    predictions = Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(add7)

    model = Model(inputs=inputs, outputs=predictions)
    return model

def Discriminator():
    lrelu = LeakyReLU(0.2)

    inputs = Input(shape=(256, 256, 3), name='input')
    x = Conv2D(64, 3, padding='same')(inputs)
    x = Activation(lrelu)(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(256, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Conv2D(512, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(lrelu)(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation(lrelu)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model

def Generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
