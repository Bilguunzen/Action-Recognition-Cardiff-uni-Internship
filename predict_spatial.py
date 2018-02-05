from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import cv2, numpy as np
import scipy.io as sio
import os
import copy
import foolbox
import paths
folder_x = open(paths.get_test_x(), "r").read().splitlines()
test_y = open(paths.get_test_y(), "r").read().splitlines()
types = [[0, 0], [12, 0], [0, 98], [12, 98], [6, 49]]
model = load_model(paths.get_spatial_model_path())
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def get_frame(path, tp, fl):
    im = cv2.imread(path)
    sx = types[tp][0]
    sy = types[tp][1]
    im = np.array(im).astype(np.float32)
    im = im[sx:sx+224,sy:sy+224,:]
    
    im[:,:,0] -= 104
    im[:,:,1] -= 116
    im[:,:,2] -= 123
    im = im[:,:,::-1]
    '''for i in range(224):
        for j in range(224):
            x = copy.copy(im[i][j])
            im[i][j][0] = x[2]
            im[i][j][1] = x[1]
            im[i][j][2] = x[0]
    '''
    if fl == True:
        im = np.fliplr(im)
    return im
 
def get_predict(path):
    prep = []
    sz = len([name for name in os.listdir(path)])
    step = int(sz / 25)
    
    for i in range(1, sz, step):
        cur_im = '/frame000'
        if i < 10:
            cur_im += '00'
        if i >= 10 and i < 100:
            cur_im += '0'
        full_path = path + cur_im + str(i) + '.jpg'
        for fl in range(2):
            for tp in range(5):
                prep.append(get_frame(full_path, tp, fl)) 
    prep = np.array(prep).astype(np.float32)
    prediction = model.predict(prep, batch_size=25, verbose=0)
    prediction = np.average(prediction, axis=0)
    
    return prediction

