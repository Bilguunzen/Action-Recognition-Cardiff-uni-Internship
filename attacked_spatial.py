from keras.models import load_model
import cv2, numpy as np
import os
import copy
import foolbox
import keras
import time
import paths
keras.backend.set_learning_phase(0)
folder_x = open(paths.get_test_x(), "r").read().splitlines()
test_y = open(paths.get_test_y(), "r").read().splitlines()


types = [[0, 0], [12, 0], [0, 98], [12, 98], [6, 49]]
model = load_model(paths.get_spatial_model_path())
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=preprocessing)
attack  = foolbox.attacks.FGSM(fmodel)

def get_frame(path, tp, fl, label):
    im = cv2.imread(path)
    #print(im.shape)
    sx = types[tp][0]
    sy = types[tp][1]
    im = np.array(im).astype(np.float32)
    im = im[sx:sx+224,sy:sy+224,:]
    if fl == True:
        im = np.fliplr(im)
    im1 = attack(im[:,:,::-1], int(label))
    if str(type(im1)) == "<class 'numpy.ndarray'>" and "(224, 224, 3)" == str(im1.shape):
        im = im1
    return im
 
def get_predict(path, label, tm):
    prep = []
    sz = len([name for name in os.listdir(path)])
    step = int(sz / 25)
    ok = False
    for i in range(1, sz, step):
        if ok == True:
            break
        cur_im = '/frame000'
        if i < 10:
            cur_im += '00'
        if i >= 10 and i < 100:
            cur_im += '0'
        full_path = path + cur_im + str(i) + '.jpg'
        for fl in range(2):
            if ok == True:
                break
            for tp in range(5):
                cur = get_frame(full_path, tp, fl, label)
                prep.append(cur)
                #if int(time.time() - tm) > 70:
                #    ok = True
                #    break
    prep = np.array(prep).astype(np.float32)
    prediction = model.predict(prep, batch_size=25, verbose=0)
    prediction = np.average(prediction, axis=0)
    
    return prediction

