import os
import numpy as np
import cv2
import gc
#from skimage.measure import compare_ssim as ssim
from datetime import datetime
from data_gen import xygen, batchgen, count_file


def Predict_BGR(model, category, pre_dir):
    path = "test_" + category

    gen = xygen(path)
    x, y = next(batchgen(gen, count_file(path)))


    pre = model.predict_on_batch(x)

    for i, img in enumerate(pre):
        out = np.array(img) * 255.0
        filename = "%s/%d.png" % (pre_dir, i)
        cv2.imwrite(filename, out)


def Predict_HS(model, category, pre_dir):
    mono_list = []
    v_list = []
    file_list = []
    org_list = []
    org_h = []
    org_s = []
    path = "test_" + category
    org_dir = "test_" + category +"_orig"
    #folder = "pre_HSV_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.mkdir(folder)

    for file in os.listdir(path):
        if file != ".DS_Store":
            filepath =  path + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            v_list.append(np.ravel(src))
            mono_list.append(np.ravel(src) / 255.0)
            file_list.append(file)


    for file in os.listdir(org_dir):
        if file != ".DS_Store":
            filepath = org_dir + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            org_list.append(src)
            org_h.append(np.ravel(src[:,:,0]))
            org_s.append(np.ravel(src[:,:,1]))

    mono_list = np.array(mono_list)
    pre = model.predict(mono_list)
    org_list = np.array(org_list)
    org_h = np.array(org_h)
    org_s = np.array(org_s)
    v_list = np.array(v_list)

    f = open(pre_dir + '/ssim.txt', 'a')



    for h, s, v, org, h_org, s_org, filename in zip(pre[0], pre[1], v_list, org_list, org_h, org_s, file_list):
        h = np.array(h*179.0, dtype='uint8')
        s = np.array(s*255.0, dtype='uint8')
        out = np.c_[h,s,v]
        out.resize((256, 256, 3))
        out = np.array(out, dtype='uint8')
        filepath = pre_dir + "/" + filename
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imwrite(filepath, out)

        h_ssim = ssim(h, h_org)
        s_ssim = ssim(s, s_org)
        pic_ssim = ssim(org, out, multichannel=True)

        f.write(filename + ':\n    h:' + str(h_ssim) + '\n    s:' + str(s_ssim) + '\n    hsv:' + str(pic_ssim) + '\n')


    f.close()

    plot_model(model, (pre_dir + "/model.png"), show_shapes=True)




def Predict_HSCOV(Hmodel, Smodel, category, pre_dir):
    mono_list = []
    v_list = []
    file_list = []
    org_list = []
    org_h = []
    org_s = []
    path = "test_" + category
    org_dir = "test_" + category +"_orig"

    for file in os.listdir(path):
        if file != ".DS_Store":
            filepath = path + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.resize(src, (128, 128))
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            v_list.append(np.ravel(src))
            src_cov = np.cov(src) / np.max(np.absolute(np.cov(src)))
            mono_list.append(np.ravel(np.append(src/255.0, src_cov)))
            file_list.append(file)

    for file in os.listdir(org_dir):
        if file != ".DS_Store":
            filepath = org_dir + "/" + file
            src = cv2.imread(filepath, 1)
            src = cv2.resize(src, (128, 128))
            src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            org_list.append(src)
            org_h.append(np.ravel(src[:,:,0]))
            org_s.append(np.ravel(src[:,:,1]))

    mono_list = np.array(mono_list)
    hpre = Hmodel.predict(mono_list)
    spre = Smodel.predict(mono_list)
    org_list = np.array(org_list)
    org_h = np.array(org_h)
    org_s = np.array(org_s)
    v_list = np.array(v_list)

    del mono_list
    gc.collect()

    f = open(pre_dir + '/ssim.txt', 'a')

    for h, s, v, org, h_org, s_org, filename in zip(hpre, spre, v_list, org_list, org_h, org_s, file_list):
        h = np.array(h*179.0, dtype='uint8')
        s = np.array(s*255.0, dtype='uint8')
        out = np.c_[h,s,v]
        out.resize((128, 128, 3))
        out = np.array(out, dtype='uint8')
        filepath = pre_dir + "/" + filename
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imwrite(filepath, out)

        h_ssim = ssim(h, h_org)
        s_ssim = ssim(s, s_org)
        pic_ssim = ssim(org, out, multichannel=True)

        f.write(filename + ':\n    h:' + str(h_ssim) + '\n    s:' + str(s_ssim) + '\n    hsv:' + str(pic_ssim) + '\n')


    f.close()

    plot_model(Hmodel, (pre_dir + "/model.png"), show_shapes=True)
