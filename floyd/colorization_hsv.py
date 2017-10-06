# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from datetime import datetime
import os
import argparse


import load_train_data
import multilayer_perceptron
import predict
#import plot

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batch', type=int, default=100)
parser.add_argument('-c', '--category', type=str, default="grass")
args = parser.parse_args()


h_train, s_train, mono_train = load_train_data.Load_hsv(args.category)

Hmodel = multilayer_perceptron.Build()
Smodel = multilayer_perceptron.Build()


batch_size = args.batch
epochs = args.epochs

Hmodel.summary()

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
#tensor_board = TensorBoard(log_dir=folder, histogram_freq=1, write_graph=True)

history_h = Hmodel.fit(mono_train, h_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])

#basename = "model/Hmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Hmodel.save(basename)
print ("\007")


#early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=1)
history_s = Smodel.fit(mono_train, s_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])

#basename = "model/Smodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Smodel.save(basename)
print ("\007")


predir = "/output/pre_HSV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(predir)
predict.Predict_HS(Hmodel, Smodel, args.category, predir)
print ("\007")
#plot.Plot_history(history_h.history, predir+"/h_history")
#plot.Plot_history(history_s.history, predir+"/s_history")

