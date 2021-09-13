#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:29:09 2021

@author: alex
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

path_hist=os.path.join(os.getcwd(),"tmp")


hist_dir={}
bpath=os.path.join(os.getcwd(),"tmp")
flist=sorted([i for i in os.listdir(bpath) if "history" in i])


for fpath in flist:
    file_path=os.path.join(bpath,fpath)
    with open(file_path,'r') as f:
        his=json.load(f)
    hist_dir[fpath]=his

def plot_learning_curves(hist,title=""):
    plt.figure()
    ax=plt.gcf().add_subplot(111)
    loss,val_loss=hist["loss"], hist["val_loss"]
    ax.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss "+title)
    ax.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss "+title)
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.suptitle(title)
    
for k in hist_dir.keys():
    plot_learning_curves(hist_dir[k],title=k)