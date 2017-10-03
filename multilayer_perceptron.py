from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers import noise
from keras.layers.core import Dropout, Activation
from keras import optimizers as op
from keras import backend as K

act = 'relu'

def Relu_advanced(x):
    return K.relu(x, max_value=1.0)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def Build():
    model = Sequential()

    inputs = Input(shape=(65536,))
    #x = noise.AlphaDropout(0.4)(inputs)
    x = Dropout(0.4)(inputs)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(200, activation=act)(x)
    #x = noise.AlphaDropout(0.4)(x)
    x = Dropout(0.4)(x)
    predictions = Dense(65536, activation='sigmoid')(x)
    model = Model(input=inputs, output=predictions)

    model.compile(loss='mean_squared_error', optimizer=op.Adam(), metrics=['mse'])

    return model

def Build_128():


    inputs = Input(shape=(32768,))
    x = Dense(500, activation=act)(inputs)
    x = Dropout(0.4)(x)
    #x = noise.AlphaDropout(0.4)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(400, activation=act)(x)
    x = Dropout(0.1)(x)
    x = Dense(500, activation=act)(x)
    x = Dropout(0.4)(x)
    #x = noise.AlphaDropout(0.4)(x)
    predictions = Dense(16384, activation='sigmoid')(x)
    model = Model(input=inputs, output=predictions)

    model.compile(loss='mean_squared_error', optimizer=op.Adam(), metrics=[mean_pred])

    return model