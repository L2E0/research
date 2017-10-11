# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
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
parser.add_argument('-s', '--steps', type=int, default=10)
args = parser.parse_args()

generator_h= load_train_data.Load_cov(args.category, args.batch, 'h')
generator_s = load_train_data.Load_cov(args.category, args.batch, 's')


Hmodel = multilayer_perceptron.Build_128()


batch_size = args.batch
epochs = args.epochs 
steps = args.steps

Hmodel.summary()

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

history_h = Hmodel.fit_generator(generator_h, 
        steps_per_epoch = steps,
        epochs=epochs,
        validation_data = validation_data_gen.Load_valid(args.category, 128, 'h', cov=True),
        validation_steps = 10,
        verbose=1,
        callbacks=[early_stopping])
#basename = "model/Hmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Hmodel.save(basename)

Smodel = multilayer_perceptron.Build_128()

history_s = Smodel.fit_generator(generator_s, 
        steps_per_epoch = steps,
        epochs=epochs,
        validation_data = validation_data_gen.Load_valid(args.category, 128, 's', cov=True),
        validation_steps = 10,
        verbose=1,
        callbacks=[early_stopping])

#basename = "model/Smodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Smodel.save(basename)


predir = "pre_HSVCOV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(predir)
predict.Predict_HSCOV(Hmodel, Smodel, args.category, predir)
plot.Plot_history(history_h.history, predir+"/h_history")
plot.Plot_history(history_s.history, predir+"/s_history")

