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
import tensorflow as tf

print(" GPU AVAILABLE : ")
print(tf.config.list_physical_devices('GPU'))

def import_data(log_path="",small_test_dataset=False):

    raw_data=pd.read_csv(log_path)
    
    print("PROCESSING DATA...")
    
    prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
    prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
    
    for i in range(3):
        prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
        
        
    prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
    prep_data['t']-=prep_data['t'][0]
    prep_data=prep_data.drop(index=[0,len(prep_data)-1])
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)//50] if small_test_dataset else prep_data
    for k in data_prepared.keys():
        if "speed" in k:
            data_prepared[k]/=10.0
        if 'acc' in k:
            data_prepared[k]/=5.0
        if 'PWM'in k: 
            data_prepared[k]=(data_prepared[k]-1500)/1000    
    
    return data_prepared

# %%% preprocess data

from tensorflow import keras

from sklearn.utils import gen_batches
import random



dyn_model=tf.keras.Sequential([keras.layers.Dense(13),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(7,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(3,activation="tanh")])

def model(vtype,lognmbr, name_tensor):
    
    log_path=os.path.join('./logs/%s/vol%s/log_real_processed.csv'%(vtype,lognmbr))     
    data_prepared=import_data(log_path,small_test_dataset=False)
    
    
    X=data_prepared[['speed[0]',
            'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
            'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
            'PWM_motor[6]']][:-1].values

    Y=np.array(data_prepared[['speed[0]','speed[1]','speed[2]']])[1:]
    
    indexes_batches=list(gen_batches(len(X),len(X)//2))
    
    idb=indexes_batches[:-1]
    random.shuffle(idb)
    
    X=np.array([X[i] for i in idb])
    Y=np.array([Y[i] for i in idb])
            
    tensor_model = keras.models.load_model("./MLmodel/"+name_tensor)
    
    speed = tensor_model.predict(X)
    return speed, Y, X
  
for i in ["speedpred_avion_log123_optimadam", "speedpred_avion_log123_optimsgd", "acc_plane"]:
    name_tensor=i
    y_pred, y_log, X=model('avion','123', name_tensor)
    
    import matplotlib.pyplot as plt
    
    fig=plt.figure()
    fig.suptitle(name_tensor)
    for i in range(3):
        fig.add_subplot(3,1,i+1)
        RMS_error =  (y_pred[:,:,0] - y_log[:,:,0])**2
        RMS_error = np.sqrt(np.mean(RMS_error))
        plt.plot(y_pred[:,:,i].flatten(), label="speed_pred["+str(i)+"]", color='red')
        plt.plot(y_log[:,:,i].flatten(), label="speed_real["+str(i)+"]", color='black')
        plt.title('RMS error : '+str(RMS_error))
        plt.ylabel('Vitesse (m/s)')
        plt.xlabel('Time')
        plt.grid()
        plt.legend()
              