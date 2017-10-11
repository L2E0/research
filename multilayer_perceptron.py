from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers import noise
from skimage.measure import compare_ssim as ssim
from keras.layers.core import Dropout, Activation
from keras import optimizers as op
from keras import backend as K
import numpy as np

act = 'relu'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def Relu_advanced(x):
    return K.relu(x, max_value=1.0)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def Build():
    model = Sequential()

    inputs = Input(shape=(65536,))
    #x = noise.AlphaDropout(0.4)(inputs)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(inputs)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    x = Dense(200,kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    #x = noise.AlphaDropout(0.4)(x)
    predictions = Dense(65536, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='mse', optimizer=op.Adam(), metrics=['mse'])

    return model

def Build_128():


    inputs = Input(shape=(32768,))
    x = Dense(4096, kernel_initializer='zeros', bias_initializer='zeros', activation=act)(inputs)
    #x = noise.AlphaDropout(0.4)(x)
    #x = Dense(2048, activation=act)(x)
    #x = Dense(2048, activation=act)(x)
    #x = Dense(2048, activation=act)(x)
    #x = Dense(2048, activation=act)(x)
    #x = Dense(2048, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    x = Dense(4096, kernel_initializer='zeros', bias_initializer='zeros', activation=act)(x)
    #x = noise.AlphaDropout(0.4)(x)
    predictions = Dense(16384, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='mse', optimizer=op.Adam(), metrics=['mae'])

    return model