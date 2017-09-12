from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation#, Dense
from keras import optimizers as op
from keras import backend as K

act = 'relu'

def Relu_advanced(x):
    return K.relu(x, max_value=1.0)

def Build():
    model = Sequential()

    #model.add(Dense(200, input_shape=(65536,)))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dense(65536))
    #model.add(Activation('sigmoid'))

    model.add(Dense(256, input_shape=(65536,)))
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation(act))
    model.add(Dropout(0.1))
    model.add(Dense(200))
    model.add(Activation(act))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Dense(65536, name='sigmoid'))
    model.add(Activation('sigmoid'))

    #model.add(Dense(100, input_shape=(65536,)))
    #model.add(Activation(act))
    #model.add(Dense(100))
    #model.add(Activation(act))
    #model.add(Dense(65536))
    #model.add(Activation('tanh'))

    #model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['mse'])

    model.compile(loss='mean_squared_logarithmic_error', optimizer=op.Adam(), metrics=['mse'])

    return model

def Build_128():

    #model = Sequential()

    #model.add(Dense(512, input_shape=(32768,)))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(400))
    #model.add(Activation(act))
    #model.add(Dense(16284))
    #model.add(Activation('sigmoid'))

    inputs = Input(shape=(32768,))
    x = Dense(500, activation=act)(inputs)
    x = Dropout(0.4)(x)
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
    #x = Dense(8192, activation=act)(inputs)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(4096, activation=act)(x)
    #x = Dense(8192, activation=act)(x)
    predictions = Dense(16384, activation='sigmoid')(x)
    model = Model(input=inputs, output=predictions)

    model.compile(loss='mean_squared_error', optimizer=op.Adam(), metrics=['mse'])

    return model