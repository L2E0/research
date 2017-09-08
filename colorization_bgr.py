# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping
from datetime import datetime

import load_train_data
import multilayer_perceptron
import predict

blue_train, green_train, red_train, mono_train = load_train_data.Load_bgr()

Bmodel = multilayer_perceptron.Build()
Gmodel = multilayer_perceptron.Build()
Rmodel = multilayer_perceptron.Build()

early_stopping = EarlyStopping(monitor='mean_squared_error', patience=3, verbose=1)

batch_size = 40
epochs = 100

Bmodel.summary()

Bmodel.fit(mono_train, blue_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Bmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Bmodel.save(basename)

Gmodel.fit(mono_train, green_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Gmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Gmodel.save(basename)

Rmodel.fit(mono_train, red_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])
basename = "model/Rmodel_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
#Rmodel.save(basename)

predict.Predict_BGR(Bmodel, Gmodel, Rmodel)


print ("\007")
