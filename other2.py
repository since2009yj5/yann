#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:32:24 2017

@author: yu
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *



data_train = pd.read_csv('train-v3.csv')
x_train = data_train.drop(['price','id'],axis=1).values  # read all column 'except' price and 'id'
#print(x_train)
#x_train = x_train.reshape(12967,21).astype('float32')


y_train = data_train['price'].values  # read price column


data_valid = pd.read_csv('valid-v3.csv')
x_valid = data_valid.drop(['price','id'],axis=1).values
y_valid = data_valid['price'].values



data_test = pd.read_csv('test-v3.csv')
x_test = data_test.drop('id',axis=1).values

from sklearn.preprocessing import *

x_train = scale(x_train)
x_valid = scale(x_valid)
x_test = scale(x_test)
"""
x_train = preprocessing.scale(x_train)
x_valid = preprocessing.scale(x_valid)
"""

#print('x_train=',x_train.shape)
#print(x_train.shape[0])


model = Sequential()
model.add(Dense(units=512,input_dim=x_train.shape[1],kernel_initializer='normal',activation='relu'))
#model.add(Dropout(rate=0.25))

model.add(Dense(units=1024,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=512,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(units=1024,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=x_train.shape[1],input_dim=32,kernel_initializer='normal',activation='relu'))


model.add(Dense(1,kernel_initializer='normal'))

print(model.summary())

model.compile(loss='MAE',optimizer='adam')

model.fit(x_train,y_train,batch_size=64,epochs=80,validation_data=(x_valid,y_valid))

Y_predict = model.predict(x_test)
np.savetxt('test.csv',Y_predict,delimiter=',')
