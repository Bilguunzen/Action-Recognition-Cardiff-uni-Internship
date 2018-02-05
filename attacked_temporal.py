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
types = [[0, 0], [12, 0], [0, 97], [12, 97], [6, 48]]
path = [paths.get_temporal_path_hor(), paths.get_temporal_path_ver()]
model = load_model(paths.get_temporal_model_path())

arr = []
for i in range(20):
    arr.append(128)

arr = np.array(arr).astype(np.int32)
preprocessing = (arr, 1)

fmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=preprocessing)
attack  = foolbox.attacks.FGSM(fmodel)

 
def get_flows(fol, start_p, tp, fl):
    
    ims = []
    for j in range(2):    
        for i in range(start_p, start_p + 10, 1):
            cur_im = '/frame000'
            if i < 10:
                cur_im += '00'
            if i >= 10 and i < 100:
                cur_im += '0'
            full_path = path[j] + fol + cur_im + str(i) + '.jpg'
            im = cv2.imread(full_path, 0) 
            sx = types[tp][0]
            sy = types[tp][1]
            im = np.array(im).astype(np.float32)
            im = im[sx:sx+224, sy:sy+224]
            if fl == True:
                im = np.fliplr(im)
            ims.append(im)
    ims = np.array(ims).astype(np.float32)
    ims = np.rollaxis(ims, 2)
    ims = np.rollaxis(ims, 2)
    return ims    

def get_predict(fol, label, tm):
    prep = []
    sz = len([name for name in os.listdir(path[0] + fol)])
    step = max(int(sz / 25), int(1))
    ok = False
    for i in range(1, sz - 9, step):
        if ok == True:
            break
        for fl in range(2):
            if ok == True:
                break
            for tp in range(5):
                cur = get_flows(fol, i, tp, fl)
                cur1 = attack(cur, int(label))
                if str(type(cur1)) == "<class 'numpy.ndarray'>" and str(cur1.shape) == "(224, 224, 20)":
                    prep.append(cur1)
                else:
                    prep.append(cur)
                #if int(time.time() - tm) > 70:
                #    ok = True
                #    break     
    prep = np.array(prep).astype(np.float32)
    prediction = model.predict(prep, batch_size=25, verbose=0)
    prediction = np.average(prediction, axis=0)
    return prediction

