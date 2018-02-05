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
import attacked_temporal as pt
import attacked_spatial as ps
import time
import paths
folder_x = open(paths.get_test_x(), "r").read().splitlines()
test_y = open(paths.get_test_y(), "r").read().splitlines()
wr = open('out_predict_attack_category.txt','w')

test_y.append(201)

count_tem = 0
count_spa = 0
count = 0
sz = len(test_y)

cur_sz = np.zeros(123).astype(np.int)

for i in range(sz - 1):
    cur_sz[int(test_y[i])] += 1

cur = 0

wr = open('out','w')
for i in range(sz - 1):
    print("Attacking video name: " + folder_x[i]) 
    start_time = time.time()
    tmp_tem = pt.get_predict(folder_x[i], test_y[i], time.time())
    tmp_spa = ps.get_predict(paths.get_spatial_path() + folder_x[i], test_y[i], time.time())
    print("--- %s seconds ---" % (time.time() - start_time))
    if str(np.argmax(tmp_tem)) == str(test_y[i]):
        count_tem += 1
        
    if str(np.argmax(tmp_spa)) == str(test_y[i]):
        count_spa += 1
    print("------------------------------------") 
    mids = []
    mids.append(tmp_tem)
    mids.append(tmp_spa)
    mids = np.array(mids).astype(np.float32)
    mids = np.average(mids, axis=0)
    print (folder_x[i] + ":")
    if str(np.argmax(mids)) == str(test_y[i]):
        count += 1
        cur += 1
        print("Two-Stream attacked Unsuccessfully")
    else:
        print("Two-Stream attacked Successfully")
    print ('%.2f' % (100.0 * (i + 1) / sz) + '% ----> (' + '%.2f' % (100.0 * count_spa / (i + 1)) + '% + ' + '%.2f' % (100.0 * count_tem / (i + 1)) + '%) = ' + '%.3f' % (100.0 * count / (i + 1)) + '%')
    if test_y[i] != test_y[i + 1]:
        print("Predict " + str(test_y[i]) + " category : ---------" + "%.2f" % (100.0 * cur / cur_sz[int(test_y[i])]) + '%')
        wr.write("Predict " + str(test_y[i]) + " category : ---------" + "%.2f" % (100.0 * cur / cur_sz[int(test_y[i])]) + '%\n')
        cur = 0

print ('result = ' + str(100.0 * count / sz))
print("Predict 100 category : ---------" + "%.2f" % (100.0 * cur / cur_sz[100]) + '%')
wr.close()
