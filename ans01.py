# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:47 2017

@author: User
"""

import pandas as pd
import numpy as np


data_1=pd.read_csv('train-v3.csv')
X_train=data_1.drop(['price','id'],axis=1).values
Y_train=data_1['price'].values

data_2=pd.read_csv('valid-v3.csv')
X_valid=data_2.drop(['price','id'],axis=1).values
Y_valid=data_2['price'].values

data_3=pd.read_csv('test-v3.csv')
X_test=data_3.drop('id',axis=1).values


def normalize(train,valid,test):
    tmp=train
    mean=tmp.mean(axis=0)
    std=tmp.std(axis=0)    
    print("tmp.shape=",tmp.shape)
    print("mean.shape=",mean.shape)
    print("std.shape=",std.shape)
    print("mean=",mean)
    print("std=",std)
    train=(train-mean)/std
    valid=(valid-mean)/std
    test=(test-mean)/std
    return train,valid,test

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)


from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(50,input_dim=21,init='normal',activation='relu'))
model.add(Dense(72,input_dim=50,init='normal',activation='relu'))
model.add(Dense(78,input_dim=72,init='normal',activation='relu'))
model.add(Dense(99,input_dim=78,init='normal',activation='relu'))
model.add(Dense(67,input_dim=99,init='normal',activation='relu'))
model.add(Dense(35,input_dim=67,init='normal',activation='relu'))
model.add(Dense(1 ,init='normal'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,Y_train,batch_size=20,nb_epoch=300,validation_data=(X_valid,Y_valid))


Y_predict=model.predict(X_test)


n=len(Y_predict)+1
for i in range(1,n):
    b= np.arange(1,n,1)   
b=np.transpose([b])
Y=np.column_stack((b,Y_predict))



np.savetxt('test.csv',Y,delimiter=',',fmt='%i')