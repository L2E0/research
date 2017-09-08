import os
import numpy as np
import cv2
from datetime import datetime
from keras.utils import plot_model


def Predict_BGR(Bmodel, Gmodel, Rmodel):
    mono_list = []
    file_list = []
    folder = "pre_RBG_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(folder)

    for file in os.listdir("test"):
        if file != ".DS_Store":
            filepath = "test/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            mono_list.append(np.ravel(src) / 255.0)
            file_list.append(file)

    mono_list = np.array(mono_list)
    bpre = Bmodel.predict(mono_list)
    gpre = Gmodel.predict(mono_list)
    rpre = Rmodel.predict(mono_list)

    for b, g, r, filename in zip(bpre, gpre, rpre, file_list):
        out = np.c_[b,g,r]
        out = out.reshape((256, 256, 3))
        out = np.array(out) * 255.0
        out = np.clip(out, 0.0, 255.0)
        filename = folder + "/" + filename
        cv2.imwrite(filename, out)

    plot_model(Bmodel, (folder + "/model.png"), show_shapes=True)

def Predict_HS(Hmodel, Smodel, folder):
    mono_list = []
    v_list = []
    file_list = []
    #folder = "pre_HSV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.mkdir(folder)

    for file in os.listdir("test"):
        if file != ".DS_Store":
            filepath = "test/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            v_list.append(np.ravel(src))
            mono_list.append(np.ravel(src) / 255.0)
            file_list.append(file)

    mono_list = np.array(mono_list)
    hpre = Hmodel.predict(mono_list)
    spre = Smodel.predict(mono_list)


    for h, s, v, filename in zip(hpre, spre, v_list, file_list):
        h = h * 180.0
        s = s * 255.0
        out = np.c_[h,s,v]
        out = out.reshape((256, 256, 3))
        out = np.array(out, dtype='uint8')
        filename = folder + "/" + filename
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imwrite(filename, out)

    plot_model(Hmodel, (folder + "/model.png"), show_shapes=True)



def Predict_HSCOV(Hmodel, Smodel):
    mono_list = []
    v_list = []
    file_list = []
    folder = "pre_HSVCOV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(folder)

    for file in os.listdir("test"):
        if file != ".DS_Store":
            filepath = "test/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.resize(src, (128, 128))
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            v_list.append(np.ravel(src))
            src_cov = np.cov(src) / np.max(np.absolute(np.cov(src)))
            mono_list.append(np.ravel(np.append(src/255.0, src_cov)))
            file_list.append(file)

    mono_list = np.array(mono_list)
    hpre = Hmodel.predict(mono_list)
    spre = Smodel.predict(mono_list)
    v_list = np.array(v_list)


    for h, s, v, filename in zip(hpre, spre, v_list, file_list):
        h = h[:(int)(len(h)/2)]
        s = s[:(int)(len(s)/2)]
        h = h * 180.0
        s = s * 255.0
        out = np.c_[h,s,v]
        out.resize((128, 128, 3))
        out = np.array(out, dtype='uint8')
        filename = folder + "/" + filename
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imwrite(filename, out)

    plot_model(Hmodel, (folder + "/model.png"), show_shapes=True)
