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

dyn_model=tf.keras.Sequential([keras.layers.Dense(13),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(7,activation="relu"),
    keras.layers.Dropout(rate=0.03),
    keras.layers.Dense(3,activation="tanh")])


class MyModel(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model=dyn_model
        
    def call(self, input_tensor):
        inputs=input_tensor[0]
        # print("inputs",inputs)
        speed,cons=tf.split(inputs[0], [3,10], axis=0)
        # print("\nspeed,cons",speed,cons)

        output=tf.reshape(speed,(1,3))

        argum=tf.reshape(tf.concat([speed,cons],0),(1,13))
        # print("\nargum",argum)
        speed_pred=tf.reshape(self.model(argum),(3,))
        # print("\nspeed_pred",speed_pred)
        
        for i in range(inputs[:-1].shape[0]):
            el=inputs[i]
            output=tf.concat([output,tf.reshape(speed_pred,(1,3))],axis=0)
            _,cons=tf.split(el, [3,10], axis=0)
            argum=tf.reshape(tf.concat([speed_pred,cons],0),(1,13))
            speed_pred=tf.reshape(self.model(argum),(3,))

        return output

import json
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import random

def train_model(vtype,lognmbr,optimizer="sgd",lr=1e-3):
    
    log_path=os.path.join('./logs/%s/vol%s/log_real_processed.csv'%(vtype,lognmbr))     
    data_prepared=import_data(log_path,small_test_dataset=False)
    
    
    X=data_prepared[['speed[0]',
            'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
            'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
            'PWM_motor[6]']][:-1].values
    print(X.shape)
    Y=np.array(data_prepared[['speed[0]','speed[1]','speed[2]']])[1:]
    
    indexes_batches=list(gen_batches(len(X),len(X)//100))
    
    idb=indexes_batches[:-1]
    random.shuffle(idb)
    
    X=np.array([X[i] for i in idb])
    Y=np.array([Y[i] for i in idb])
    
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,shuffle=False)
    print(X_train.shape)

    model = MyModel()
        
    optisgd=tf.keras.optimizers.SGD(learning_rate=lr)
    optiadam=tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(loss="mean_squared_error",
                  optimizer=optisgd if optimizer=="sgd" else optiadam,
                      metrics=[tf.keras.metrics.MeanSquaredError()])
    
    checkpoint_filepath = './tmp/checkpoint_%s_log%s_optim%s'%(vtype,lognmbr,optimizer)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    history = model.fit(X_train, y_train, 
                        epochs=50,
                        batch_size=1,
                        shuffle=True,
                        callbacks=[model_checkpoint_callback],
                        validation_data=(X_test,y_test)) 


    history_dict = history.history
    your_history_path="./tmp/history_%s_log%s_optim%s"%(vtype,lognmbr,optimizer)
    json.dump(history_dict, open(your_history_path, 'w'))
    import shutil 
    dir_name="./MLmodel/speedpred_%s_log%s_optim%s"%(vtype,lognmbr,optimizer)
    try:
        shutil.rmtree("./%s"%(dir_name))
    except:
        pass
    os.makedirs("./%s"%(dir_name))
    tf.saved_model.save(dyn_model,"./%s"%(dir_name))
    return True


lr_copt_sgd=0.008
lr_copt_adam=0.0001
lr_avi_sgd=0.0005
lr_avi_adam=0.00005


# train_model('avion','1',optimizer="sgd",lr=lr_avi_sgd)
# train_model('avion','2',optimizer="sgd",lr=lr_avi_sgd)

# train_model('avion','3',optimizer="sgd",lr=lr_avi_sgd)
train_model('avion','123',optimizer="sgd",lr=lr_avi_sgd)

# train_model('copter','1',optimizer="sgd",lr=lr_copt_sgd)
# train_model('copter','2',optimizer="sgd",lr=lr_copt_sgd)

train_model('copter','12',optimizer="sgd",lr=lr_copt_sgd)


# train_model('avion','1',optimizer="adam",lr=lr_avi_adam)
# 
# train_model('avion','2',optimizer="adam",lr=lr_avi_adam)
# train_model('avion','3',optimizer="adam",lr=lr_avi_adam)

train_model('avion','123',optimizer="adam",lr=lr_avi_adam)
# train_model('copter','1',optimizer="adam",lr=lr_copt_adam)

# train_model('copter','2',optimizer="adam",lr=lr_copt_adam)
train_model('copter','12',optimizer="adam",lr=lr_copt_adam)
