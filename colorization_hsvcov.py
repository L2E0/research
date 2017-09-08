# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from datetime import datetime

import load_train_data
import multilayer_perceptron
import predict

h_train, s_train, mono_train = load_train_data.Load_cov()

Hmodel = multilayer_perceptron.Build_128()
Smodel = multilayer_perceptron.Build_128()


batch_size = 100
epochs = 500 

Hmodel.summary()

early_stopping = EarlyStopping(monitor='mean_squared_error', patience=20, verbose=1)
Hmodel.fit(mono_train, h_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
#basename = "model/Hmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Hmodel.save(basename)
print ("\007")


#early_stopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=1)
Smodel.fit(mono_train, s_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
#basename = "model/Smodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Smodel.save(basename)
print ("\007")


predict.Predict_HSCOV(Hmodel, Smodel)

print ("\007")