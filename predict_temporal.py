from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from random import randint
import cv2, numpy as np
import scipy.io as sio
import os
import paths

folder_x = open(paths.get_test_x(), "r").read().splitlines()
test_y = open(paths.get_test_y(), "r").read().splitlines()
types = [[0, 0], [12, 0], [0, 97], [12, 97], [6, 48]]
path = [paths.get_temporal_path_hor(), paths.get_temporal_path_ver()]
model = load_model(paths.get_temporal_model_path())
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
 
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
            im -= 128
            if fl == True:
                im = np.fliplr(im)
            ims.append(im)
    ims = np.array(ims).astype(np.float32)
    ims = np.rollaxis(ims, 2)
    ims = np.rollaxis(ims, 2)
    return ims    

def get_predict(fol):
    prep = []
    sz = len([name for name in os.listdir(path[0] + fol)])
    step = max(int(sz / 25), int(1))
    for i in range(1, sz - 9, step):
        for fl in range(2):
            for tp in range(5):
                prep.append(get_flows(fol, i, tp, fl))
    prep = np.array(prep).astype(np.float32)
    prediction = model.predict(prep, batch_size=25, verbose=0)
    prediction = np.average(prediction, axis=0)
    return prediction
