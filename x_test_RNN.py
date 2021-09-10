#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:54:09 2021

@author: alex
"""

# %% import data
import pandas as pd
import numpy as np
import os


def import_data(logpath="",small_test_dataset=True):

    
    raw_data=pd.read_csv(log_path)
    
    print("PROCESSING DATA...")
    
    
    prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
    prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
    
    
    # on choisit tmin et tmax selon quel log on utilise
    
    # if "vol12" in log_path:
    #     tmin,tmax=(-1,1e10) 
    # elif "vol1" in log_path:
    #     tmin,tmax=(41,265) 
    # elif "vol2" in log_path:
    #     tmin,tmax=(10,140) 
    # tmin=1
    # tmax=500
        
    # prep_data=prep_data[prep_data['t']>tmin]
    # prep_data=prep_data[prep_data['t']<tmax]
    # prep_data=prep_data.reset_index().drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
    
    
    for i in range(3):
        prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
        
        
    prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
    prep_data['t']-=prep_data['t'][0]
    prep_data=prep_data.drop(index=[0,len(prep_data)-1])
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)//50] if small_test_dataset else prep_data
    return data_prepared


# %% simple feedforward model
# %%% preprocess data

log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
data_prepared=import_data(log_path,small_test_dataset=False)

for k in data_prepared.keys():
    if "speed" in k:
        data_prepared[k]/=25.0
    if 'acc' in k:
        data_prepared[k]/=20.0
    if 'PWM'in k: 
        data_prepared[k]=(data_prepared[k]-1500)/1000

X_train_full=data_prepared[['speed[0]',
       'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
       'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
       'PWM_motor[6]']]

Y_train_full=data_prepared[['acc[0]','acc[1]','acc[2]']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=42)

# %%% preprocess data
import tensorflow as tf
from tensorflow import keras 

copter_model=tf.keras.Sequential([keras.Input(shape=(13,)),
    keras.layers.Dense(9),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(7),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(5),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(3)])

copter_model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["accuracy"])


import datetime

log_dir = "./tfres/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = copter_model.fit(X_train, y_train, epochs=10,callbacks=[tensorboard_callback]) 
acc_pred=copter_model(X_train_full)                        

 
import matplotlib.pyplot as plt

plt.figure()
for i in range(3):
    
    ax=plt.gcf().add_subplot(3,1,i+1)
    ax.plot(data_prepared['t'],data_prepared['acc[%i]'%(i)],color="black",label="data")
    ax.plot(data_prepared['t'],data_prepared['acc_ned_grad[%i]'%(i)],color="blue",label="data",alpha=0.5)

    ax.plot(data_prepared['t'][np.arange(len(acc_pred))],acc_pred[:,i],color="red",label="pred")
    plt.grid()
    
    
# %% speed prediction feedworward
# %%% preprocess data

log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
data_prepared=import_data(log_path,small_test_dataset=False)

for k in data_prepared.keys():
    if "speed" in k:
        data_prepared[k]/=25.0
    if 'acc' in k:
        data_prepared[k]/=20.0
    if 'PWM'in k: 
        data_prepared[k]=(data_prepared[k]-1500)/1000

X_train_full=data_prepared[['speed[0]',
       'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
       'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
       'PWM_motor[6]']][:-1]

Y_train_full=data_prepared[['speed[0]','speed[1]','speed[2]']][1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=42)

# %%% preprocess data
import tensorflow as tf
from tensorflow import keras 

copter_model=tf.keras.Sequential([keras.Input(shape=(13,)),
    keras.layers.Dense(9),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(7),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(5),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(3)])

copter_model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["accuracy"])


import datetime

log_dir = "./tfres/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = copter_model.fit(X_train, y_train, epochs=10,callbacks=[tensorboard_callback]) 
acc_pred=copter_model(X_train_full)                        

 
import matplotlib.pyplot as plt

plt.figure()
for i in range(3):
    
    ax=plt.gcf().add_subplot(3,1,i+1)
    ax.plot(data_prepared['t'],data_prepared['speed[%i]'%(i)],color="black",label="data")

    ax.plot(data_prepared['t'][np.arange(len(acc_pred))],acc_pred[:,i],color="red",label="pred")
    plt.grid()
