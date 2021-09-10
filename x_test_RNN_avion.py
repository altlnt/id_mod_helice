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
    
    for i in range(3):
        prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
        
        
    prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
    prep_data['t']-=prep_data['t'][0]
    prep_data=prep_data.drop(index=[0,len(prep_data)-1])
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)//50] if small_test_dataset else prep_data
    for k in data_prepared.keys():
        if "speed" in k:
            data_prepared[k]/=25.0
        if 'acc' in k:
            data_prepared[k]/=20.0
        if 'PWM'in k: 
            data_prepared[k]=(data_prepared[k]-1500)/1000    
    
    return data_prepared

def plot_learning_curves(ax,hist):
    loss,val_loss=history.history["loss"], history.history["val_loss"]
    ax.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    ax.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    ax.legend(fontsize=14)
    ax.grid(True)
    
# %% SIMPLE feedforward model: ACC
    # %%% preprocess data

log_path=os.path.join('./logs/avion/vol123/log_real_processed.csv')     
data_prepared=import_data(log_path,small_test_dataset=False)


X_train_full=data_prepared[['speed[0]',
       'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
       'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
       'PWM_motor[6]']]

Y_train_full=data_prepared[['acc[0]','acc[1]','acc[2]']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=42)

    # %%% feedforward model
import tensorflow as tf
from tensorflow import keras 

copter_model=tf.keras.Sequential([keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(13,activation="relu"),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(13),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(7),
    keras.layers.Dropout(rate=0.05),
    keras.layers.Dense(3,activation="tanh")])


copter_model.compile(loss="mean_squared_error",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.MeanSquaredError()])



history = copter_model.fit(X_train, y_train, epochs=20,validation_data=(X_test,y_test)) 

    # %%% pred and plot

acc_pred=copter_model.predict(X_train_full)                        



import matplotlib.pyplot as plt

plt.figure()
for i in range(3):
    
    ax=plt.gcf().add_subplot(3,2,2*i+1)
    ax.plot(data_prepared['t'],data_prepared['acc[%i]'%(i)],color="black",label="data")
    ax.plot(data_prepared['t'],data_prepared['acc_ned_grad[%i]'%(i)],color="blue",label="data",alpha=0.5)

    ax.plot(data_prepared['t'][np.arange(len(acc_pred))],acc_pred[:,i],color="red",label="pred")
    plt.grid()
    
ax=plt.gcf().add_subplot(1,2,2)
plot_learning_curves(ax,history)

# # %% SIMPLE feedforward model: SPEED
#     # %%% preprocess data

# log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
# data_prepared=import_data(log_path,small_test_dataset=False)


# X_train_full=data_prepared[['speed[0]',
#        'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
#        'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
#        'PWM_motor[6]']][:-1]

# Y_train_full=data_prepared[['speed[0]','speed[1]','speed[2]']][1:]

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=42)

#     # %%% feedforwad model
# import tensorflow as tf
# from tensorflow import keras 

# copter_model=tf.keras.Sequential([keras.layers.Dense(13,activation="tanh"),
#     keras.layers.Dropout(rate=0.05),
#     keras.layers.Dense(7,activation="tanh"),
#     keras.layers.Dropout(rate=0.05),
#     keras.layers.Dense(5,activation="tanh"),
#     keras.layers.Dropout(rate=0.05),
#     keras.layers.Dense(3,activation="tanh")])

# copter_model.compile(loss="mean_squared_error",
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#                   metrics=[tf.keras.metrics.MeanSquaredError()])

# history = copter_model.fit(X_train, y_train, 
#                            epochs=15,
#                            validation_data=(X_test,y_test)) 


#     # %%% pred plot


# v_pred_no_iter=copter_model.predict(X_train_full)


# v_pred=[np.array([data_prepared['speed[%i]'%(i)][0] for i in range(3)])]
        
# for i in X_train_full.index:
#     print("\r Pred on batch %i / %i "%(i,max(X_train_full.index)), end='', flush=True)

#     x=X_train_full.loc[i]
    
#     for j in range(3):
#         x['speed[%i]'%(j)]=v_pred[-1][j]



#     new_v=copter_model.predict(x.values.reshape(-1,13))
#     v_pred.append(new_v.reshape(3,-1))

# v_pred[0]=v_pred[0].reshape(3,-1)     
# v_pred_stack=np.array(v_pred).reshape(-1,3)              

# import matplotlib.pyplot as plt

# plt.figure()
# for i in range(3):
    
#     ax=plt.gcf().add_subplot(3,1,i+1)
#     ax.plot(data_prepared['t'],data_prepared['speed[%i]'%(i)],color="black",label="data")

#     ax.plot(data_prepared['t'][np.arange(len(v_pred_stack))],v_pred_stack[:,i],color="red",label="pred")
#     ax.plot(data_prepared['t'][np.arange(len(v_pred_no_iter))],v_pred_no_iter[:,i],color="green",label="pred no iter")

#     plt.grid()
# # %% speed pred recurrent
#     # %%% preprocess data

# log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
# data_prepared=import_data(log_path,small_test_dataset=False)

# n_steps=10

# X=data_prepared[['speed[0]',
#        'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'PWM_motor[1]',
#        'PWM_motor[2]', 'PWM_motor[3]', 'PWM_motor[4]', 'PWM_motor[5]',
#        'PWM_motor[6]']][:-n_steps].values

# # X=X.reshape(X.shape[0],1,1,X.shape[1])
# X=X.reshape(X.shape[0],1,X.shape[1])

# Y=np.array([data_prepared[['speed[0]','speed[1]','speed[2]']][:n_steps].values])
 
# for i in range(1,len(X)):
#     Y=np.concatenate((Y,[data_prepared[['speed[0]','speed[1]','speed[2]']][i:i+n_steps].values]),axis=0)

#     # %%% recurrent model



# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# class CustomNet(keras.models.Model):
#     def __init__(self, init_speed=np.zeros(3), **kwargs):
#         super().__init__(**kwargs)
#         self.innermodel=tf.keras.Sequential([keras.layers.Dense(13,activation="tanh"),
#             keras.layers.Dropout(rate=0.05),
#             keras.layers.Dense(7,activation="tanh"),
#             keras.layers.Dropout(rate=0.05),
#             keras.layers.Dense(5,activation="tanh"),
#             keras.layers.Dropout(rate=0.05),
#             keras.layers.Dense(3,activation="tanh")])

#     def call(self, inputs):
        
#         first_pred=self.innermodel(inputs)
        
#         for _ in range(1,len(model)):
#             Z = self.block1(Z)
#         Z = self.block2(Z)
#         return self.out(Z)

# rnn_model=keras.models.Sequential([
#     keras.layers.SimpleRNN(3, return_sequences=True,  input_shape=[ 1,13]),
#     # keras.layers.SimpleRNN(3, return_sequences=True,input_shape=[20])
#     keras.layers.SimpleRNN(3, return_sequences=True)
#     ])


# rnn_model.compile(loss="mse",
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))


# # import datetime

# # log_dir = "./tfres/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train_ds= tf.data.Dataset.from_tensor_slices((X_train,y_train))
# test_ds= tf.data.Dataset.from_tensor_slices((X_test,y_test))

# history = rnn_model.fit(X_train, y_train,
#                         epochs=1) 
# print(rnn_model.summary())
#     # %%% pred and plot
# speed_pred=rnn_model.predict(X[0].reshape(-1,1,13))                        
# print(speed_pred)


# speed_pred=np.concatenate((Y[:10],speed_pred),axis=1)

# import matplotlib.pyplot as plt

# plt.figure()
# for i in range(3):
    
#     ax=plt.gcf().add_subplot(3,1,i+1)
#     ax.plot(data_prepared['t'],data_prepared['speed[%i]'%(i)],color="black",label="data")

#     # ax.plot(data_prepared['t'][np.arange(len(v_pred_stack))],v_pred_stack[:,i],color="red",label="pred")
#     ax.plot(data_prepared['t'][np.arange(len(speed_pred))],speed_pred[:,i],color="green",label="pred no iter")

#     plt.grid()








