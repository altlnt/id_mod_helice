#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 23:45:53 2021

@author: alex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json


base_path=os.path.join(os.getcwd(),"results")

# %%  PLOT ONE OPTI
opti_to_plot="5s_struc_all"

def detect_nans(x,y):
    nx,ny=[],[]
    for j,i in enumerate(y):
        if np.isnan(i):
            nx.append(x[j])
            ny.append(i)
            
    return nx,ny
    

def plot_opti(opti_to_plot):
    
    print("Processing %s"%(opti_to_plot))
    dirpath=os.path.join(base_path,opti_to_plot)
    
    "listing all json files except data.json which has the metaparams"
    json_list=[]
    
    for file in os.listdir(dirpath):
        json_list.append(file) if file!="data.json" else None
        
    if len(json_list)==0:
        return print("EMPTY LOG...")
    
    "parsing meta_data json"
    with open(os.path.join(dirpath,"data.json"),'r') as f:
      data = json.load(f)        
          
    opti_params={}
    param_key_list=('log_path',
                    'fit_on_v',
                    'used_logged_v_in_model',
                    'used_logged_v_in_model',
                    'base_lr',
                    'nsecs',
                    'vanilla_force_model',
                    'structural_relation_idc1',
                    'structural_relation_idc2',
                    'assume_nul_wind',
                    'di_equal_dj')
    
    for i in param_key_list:
        opti_params[i]=data[i]
        
    textstr='\n'.join(["%s=%s"%(i,opti_params[i]) for i in opti_params])

    
    
    # print(json_list)
    "constructing key list"
    for file in json_list:
        if ("epoch" in file) or ("start" in file):
            print("Epoch data found...")
            with open(os.path.join(dirpath,file),"r") as f:
                data=json.load(f)
                keys_=data.keys()
            break
    
    "here we declare the data dict structure from these keys"
    data_dict={}
    
    for k in keys_:
        data_dict[k]=[]
        
    for file in json_list:
        with open(os.path.join(dirpath,file),"r") as f:
            json_data=json.load(f)
            for k in keys_:
                data_dict[k]=data_dict[k]+[json_data[k]] if isinstance(json_data[k],(int,float)) else data_dict[k]+json_data[k] 
    # print(data_dict)
    
    N_sp=len(keys_)
    # data_df=pd.DataFrame(data=[data_dict[i] for i in data_dict],columns=data_dict.keys())
    f=plt.figure()
    f.suptitle("NAME=%s"%(opti_to_plot))
    score_ax=f.add_subplot(2,3,2)
    
    
    "display scores "
    
    
    y0=pd.DataFrame(data=data_dict['train_sc_a'],columns=["train_sc_a"]).fillna(0)
    y1=pd.DataFrame(data=data_dict['val_sc_a'],columns=["val_sc_a"]).fillna(0)
    y2=pd.DataFrame(data=data_dict['total_sc_a'],columns=["total_sc_a"]).fillna(0)
    
    y0=y0.loc[y0['train_sc_a']>0]
    y1=y1.loc[y1['val_sc_a']>0]
    y2=y2.loc[y2['total_sc_a']>0]
    
    
    # xnan0,y0nans=detect_nans(range(len(y0)), y0)
    # xnan1,y1nans=detect_nans(range(len(y1)), y1)
    # xnan2,y2nans=detect_nans(range(len(y2)), y2)
    
    x0,x1,x2=y0.index,y1.index,y2.index
    
    score_ax.plot(x0,y0,label="train_sc_a")
    score_ax.plot(x1,y1,label="val_sc_a")
    score_ax.plot(x2,y2,label="total_sc_a",marker="o")


    score_ax.grid(),score_ax.legend(),score_ax.set_ylim(0,2),score_ax.set_xlim(0,)
    
    score_vx=f.add_subplot(2,3,5)

    y0=pd.DataFrame(data=data_dict['train_sc_v'],columns=["train_sc_v"]).fillna(0)
    y1=pd.DataFrame(data=data_dict['val_sc_v'],columns=["val_sc_v"]).fillna(0)
    y2=pd.DataFrame(data=data_dict['total_sc_v'],columns=["total_sc_v"]).fillna(0)
    
    y0=y0.loc[y0['train_sc_v']>0]
    y1=y1.loc[y1['val_sc_v']>0]
    y2=y2.loc[y2['total_sc_v']>0]
    
    score_vx.plot(x0,y0,label="train_sc_v")
    score_vx.plot(x1,y1,label="val_sc_v")
    score_vx.plot(x2,y2,label="total_sc_v",marker="o")


    score_vx.grid(),score_vx.legend(),score_vx.set_ylim(0,2),score_vx.set_xlim(0,)
    
    " display opti metaparams "
    
    ax_box=f.add_subplot(2,3,3)
    ax_box.remove()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    score_ax.text(0.05, 0.95, 
                  textstr, transform=ax_box.transAxes, 
                  fontsize=9,
        verticalalignment='top',horizontalalignment="left", bbox=props)
    
    
    
    
    
    
    
    j=0
    regroup=[["di","dj","dk"],["vw_i","vw_j"]]
    # print(np.array([regroup],dtype="object").flatten())
    for i,k in enumerate(data.keys()):
        if ('train_sc' in k) or  ("val_sc" in k) or ('total_sc' in k):
            pass
        
        elif k in [item for sublist in regroup for item in sublist]:
            pass
        
        else:   
            
            temp_ax=f.add_subplot(N_sp-6-len(regroup)-1,3,3*j+1)
            y=np.nan_to_num(data_dict[k],nan=0)
            x=np.arange(len(y))
            temp_ax.plot(x,y,label=k)
            temp_ax.legend(),temp_ax.grid(),temp_ax.set_xlim(0,)
            j+=1
            # print(j,k)
    

    for tup in regroup:
        temp_ax=f.add_subplot(N_sp-6-len(regroup)-1,3,3*j+1)

        for k in tup:
            y=np.nan_to_num(data_dict[k],nan=-1)
            x=np.arange(len(y))
            temp_ax.plot(x,y,label=k)
        temp_ax.legend(),temp_ax.grid(),temp_ax.set_xlim(0,)
        j+=1

# plot_opti(opti_to_plot)


# %%  PLOT ALL optis

list_optis=os.listdir("./results/")
# list_optis=['fit_v_True_lr_0.001000_ns_-1.000000']
[plot_opti(i) for i in list_optis]
# plot_opti('10ep_fit_v_False_lr_scipy_ns_2')
