# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:59:05 2020

@author: Debatosh
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
'''


import keras.backend as K'''





data = pd.read_csv("C:/Users/debat/Documents/Datasets/data.csv")
data_test = pd.read_csv("C:/Users/debat/Documents/Datasets/DHDD_CSV-master/test.csv")
def dataset(data):
    dataset  = np.array(data)
    np.random.shuffle(dataset)
    
    X= dataset[:, 1:1025]
    X = X/255
    Y = dataset[:, 0]
    
    print(X.shape, Y.shape)
    Y = Y.reshape(Y.shape[0], 1)
    print(X.shape, Y.shape)
    Y = Y.T
    print(X.shape, Y.shape)
    return X,Y

X,Y = dataset(data)

X_t, Y_t = dataset(data_test)


image_x = 32
image_y = 32

train_y = np_utils.to_categorical(Y)
print(X.shape, train_y.shape) 
'''njhbbhjnkmkmkmkmkkk jnn'''
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
X = X.reshape(X.shape[0], image_x, image_y, 1)

print(X.shape, train_y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,train_y, test_size = 0.2, random_state = 2)
print(X_train.shape, Y_train.shape)




def keras_model(image_x, image_y):
    num_of_classes = 10
    model = Sequential()
    model.add(Conv2D(filters= 32, kernel_size = (5,5),input_shape = (image_x, image_y, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
    model.add(Conv2D(64,(5,5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    filepath = "C:/Users/debat/Documents/Python/DeepLearning/devnagiri.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list
    
model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, Y_train, validation_data = (X_test, Y_test ), epochs = 4, batch_size = 64, callbacks = callbacks_list)
scores = model.evaluate(X_test, Y_test, verbose = 0)
print('CNN Error: ' % (100 - scores[1] * 100))
print_summary(model)
model.save('devnagiri script.h5')
    
    

    
    
    

