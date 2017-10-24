# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from datetime import datetime
import os
import argparse


import load_train_data
import model
import predict
import plot
import validation_data_gen

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batch', type=int, default=32)
parser.add_argument('-c', '--category', type=str, default="grass")
parser.add_argument('-s', '--steps', type=int, default=100)
args = parser.parse_args()


generator= load_train_data.Load_hsv(args.category, args.batch)


mlp = model.mlp()


batch_size = args.batch
epochs = args.epochs
steps = args.steps

mlp.summary()

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

history = mlp.fit_generator(
        generator, 
        steps_per_epoch = steps,
        epochs=epochs,
        validation_data = validation_data_gen.Load_valid(args.category, 256, 'hs'),
        validation_steps = 10,
        verbose=1,
        #callbacks=[early_stopping])
)



predir = "pre_HSV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(predir)
predict.Predict_HS(mlp, args.category, predir)
plot.Plot_history(history.history, predir+"/history")

basename = predir + "/mlp.h5"
mlp.save_weights(basename)
os.system('shutdown -s')
