# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from datetime import datetime
import os
import argparse

import load_train_data
import multilayer_perceptron
import predict
import plot

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batch', type=int, default=100)
parser.add_argument('-c', '--category', type=str, default="grass")
args = parser.parse_args()

blue_train, green_train, red_train, mono_train = load_train_data.Load_bgr(args.category)

Bmodel = multilayer_perceptron.Build()
Gmodel = multilayer_perceptron.Build()
Rmodel = multilayer_perceptron.Build()

early_stopping = EarlyStopping(monitor='mean_squared_error', patience=3, verbose=1)

batch_size = args.batch
epochs = args.epochs

Bmodel.summary()

history_b = Bmodel.fit(mono_train, blue_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Bmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Bmodel.save(basename)

history_g = Gmodel.fit(mono_train, green_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Gmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Gmodel.save(basename)

history_r = Rmodel.fit(mono_train, red_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Rmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Rmodel.save(basename)

predir = "pre_BGR_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(predir)
predict.Predict_BGR(Bmodel, Gmodel, Rmodel, args.category, predir)
plot.Plot_history(history_b.history, predir+"/b_history")
plot.Plot_history(history_g.history, predir+"/g_history")
plot.Plot_history(history_r.history, predir+"/r_history")


print ("\007")
