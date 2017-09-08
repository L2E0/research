from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers as op
from keras import backend as K

act = 'relu'

def Relu_advanced(x):
    return K.relu(x, max_value=1.0)

def Build():
    model = Sequential()

    model.add(Dense(200, input_shape=(65536,)))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(200))
    model.add(Activation(act))
    model.add(Dense(65536))
    model.add(Activation('sigmoid'))

    #model.add(Dense(256, input_shape=(65536,)))
    #model.add(Activation(act))
    #model.add(Dropout(0.5))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dropout(0.1))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dropout(0.1))
    #model.add(Dense(256))
    #model.add(Activation(act))
    #model.add(Dropout(0.5))
    #model.add(Dense(65536, name='sigmoid'))
    #model.add(Activation('sigmoid'))

    #model.add(Dense(100, input_shape=(65536,)))
    #model.add(Activation(act))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dense(200))
    #model.add(Activation(act))
    #model.add(Dense(100))
    #model.add(Activation(act))
    #model.add(Dense(65536))
    #model.add(Activation('tanh'))

    #model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['mse'])

    model.compile(loss='mean_squared_error', optimizer=op.Adam(), metrics=['mse'])

    return model

def Build_128():

    model = Sequential()

    model.add(Dense(400, input_shape=(32768,)))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(400))
    model.add(Activation(act))
    model.add(Dense(32768))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=op.Adam(), metrics=['mse'])

    return model