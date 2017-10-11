# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from datetime import datetime
import os
import argparse


import load_train_data
import multilayer_perceptron
import predict
import plot
import validation_data_gen

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batch', type=int, default=100)
parser.add_argument('-c', '--category', type=str, default="grass")
args = parser.parse_args()


generator_h= load_train_data.Load_hsv(args.category, args.batch, 'h')
generator_s = load_train_data.Load_hsv(args.category, args.batch, 's')


Hmodel = multilayer_perceptron.Build()
Smodel = multilayer_perceptron.Build()


batch_size = args.batch
epochs = args.epochs

Hmodel.summary()

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

history_h = Hmodel.fit_generator(generator_h, 
        steps_per_epoch = 10,
        epochs=epochs,
        validation_data = validation_data_gen.Load_valid(args.category, 256, 'h'),
        validation_steps = 10,
        verbose=1,
        callbacks=[early_stopping])

#basename = "model/Hmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Hmodel.save(basename)
#print ("\007")


#early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=1)
history_s = Smodel.fit_generator(generator_s, 
        steps_per_epoch = 10,
        epochs=epochs,
        validation_data = validation_data_gen.Load_valid(args.category, 256, 's'),
        validation_steps = 10,
        verbose=1,
        callbacks=[early_stopping])

#basename = "model/Smodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Smodel.save(basename)
#print ("\007")


predir = "pre_HSV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(predir)
predict.Predict_HS(Hmodel, Smodel, args.category, predir)
plot.Plot_history(history_h.history, predir+"/h_history")
plot.Plot_history(history_s.history, predir+"/s_history")
