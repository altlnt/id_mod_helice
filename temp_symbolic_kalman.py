#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:34:26 2021

@author: alex
"""
import sys
import numpy as np
# %% Control Booleans
"regresion on a"
fit_on_v=fit_arg
used_logged_v_in_model=not fit_arg


model_motor_dynamics=True



log_name="stochast_fit_v_%s_lr_%s_ns_%s"%(str(fit_on_v),str(base_lr) if blr!="scipy" else 'scipy',str(ns))


with_ct3=False
vanilla_force_model=False

structural_relation_idc1=False
structural_relation_idc2=False

if vanilla_force_model and (structural_relation_idc1 or structural_relation_idc2):
    print("INVALID :")
    print("vanilla force model and structural_relation cannot be true at the same time")
    sys.exit()
    
if structural_relation_idc1 and structural_relation_idc2:
    print("INVALID :")
    print("structural_relation_idc1 and structural_relation_idc2 cannot be true at the same time")
    sys.exit()

assume_nul_wind=False
approx_x_plus_y=False
di_equal_dj=False


# ID coeffs
id_mass=False
id_blade_coeffs=True
id_c3=with_ct3
id_blade_geom_coeffs=False
id_body_liftdrag_coeffs=True
id_wind=not assume_nul_wind
id_time_const=model_motor_dynamics




# Log path

# log_path="/logs/vol1_ext_alcore/log_real.csv"
# log_path="/logs/vol1_ext_alcore/log_real.csv"
log_path="/logs/vol12/log_real_processed.csv"



mass=369 #batterie
mass+=1640-114 #corps-carton
mass/=1e3
Area=np.pi*(11.0e-02)**2
r0=11e-02
rho0=1.204
kv_motor=800.0
pwmmin=1075.0
pwmmax=1950.0
U_batt=16.8

b10=14.44

c10=0.018828
c20=0.106840
c30=0.0
ch10=0.1
ch20=0.1
di0=1.0
dj0=1.00
dk0=1.0
vwi0=0.0
vwj0=0.0
kt0=5.0

physical_params=[mass,
Area,
r0,
rho0,
kv_motor,
pwmmin,
pwmmax,
U_batt,
b10,
c10,
c20,
c30,
ch10,
ch20,
di0,
dj0,
dk0,
vwi0,
vwj0,
kt0]

bounds={}
bounds['m']=(0,np.inf)
bounds['A']=(0,np.inf)
bounds['r']=(0,np.inf)
bounds['c1']=(0,np.inf)
bounds['c2']=(-np.inf,np.inf)
bounds['c3']=(-np.inf,np.inf)
bounds['ch1']=(-np.inf,np.inf)
bounds['ch2']=(-np.inf,np.inf)
bounds['di']=(0,np.inf)
bounds['dj']=(0,np.inf)
bounds['dk']=(0,np.inf)
bounds['vw_i']=(-15,15)
bounds['vw_j']=(-15,15)
bounds['kt']=(0,np.inf)

"scaler corresponds roughly to the power of ten of the parameter"
"it does not have to though, it may be used to improve the grad descent"




metap={"model_motor_dynamics":model_motor_dynamics,
        "used_logged_v_in_model":used_logged_v_in_model,
        "with_ct3":with_ct3,
        "vanilla_force_model":vanilla_force_model,
        "structural_relation_idc1":structural_relation_idc1,
        "structural_relation_idc2":structural_relation_idc2,
        "fit_on_v":fit_on_v,
        "assume_nul_wind":assume_nul_wind,
        "approx_x_plus_y":approx_x_plus_y,
        "di_equal_dj":di_equal_dj,
        "log_path":log_path,
        "base_lr":base_lr,
        "grad_autoscale":grad_autoscale,
        "n_epochs":n_epochs,
        "nsecs":nsecs,
        "train_proportion":train_proportion,
        "[mass,Area,r,rho,kv_motor,pwmmin,pwmmax,U_batt,b1,c10,c20,c30,ch10,ch20,di0,dj0,dk0,vwi0,vwj0,kt0]":physical_params,
        "bounds":bounds}



print(" META PARAMS \n")
[print(i,":", metap[i]) for i in metap.keys()]
    
# %%   ####### Saving utility
import os
import json
import datetime
import time
if log_name=="":
    log_name=str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    
spath=os.path.join(os.getcwd(),"results",log_name)

os.makedirs(spath)

with open(os.path.join(spath,'data.json'), 'w') as fp:
    json.dump(metap, fp)

import pandas as pd
def saver(name=None,save_path="/home/alex/Documents/identification_modele_hélice/results/",**kwargs):
    
    D={}
    dname=str(int(time.time())) if (name is None) else name
    
    for i in kwargs:
        if type(kwargs[i])==dict:
            for j in kwargs[i]:
                D[j]=kwargs[i][j]
        else:
            D[i]=kwargs[i].tolist() if isinstance(kwargs[i],np.ndarray) else kwargs[i]
    
    with open(os.path.join(save_path,'%s.json'%(dname)), 'w') as fp:
        json.dump(D, fp)
        
# %% SYMPY
from sympy import *

dt=symbols('dt',positive=True,real=True)
m=symbols('m',reals=True,positive=True)


m_s=m

r1,r2,r3,r4,r5,r6,r7,r8,r9=symbols("r1,r2,r3,r4,r5,r6,r7,r8,r9",real=True)
R=Matrix([[r1,r2,r3],
          [r4,r5,r6],
          [r7,r8,r9]])


vlog_i,vlog_j,vlog_k=symbols("vlog_i,vlog_j,vlog_k",real=True)


v_i,v_j,v_k=(vlog_i,vlog_j,vlog_k) 


vw_i,vw_j=symbols('vw_i,vw_j',real=True)

v=Matrix([[v_i],
         [v_j],
         [v_k]])

if not assume_nul_wind:
    va_NED=Matrix([[v_i-vw_i],
                    [v_j-vw_j],
                    [v_k]]) 
else :
    va_NED=Matrix([[v_i],
                   [v_j],
                   [v_k]])

va_body=R.T@va_NED

k_vect=Matrix([[0],
            [0],
            [1.0]])


"motor dynamics"
t1=time.time()
print("Elapsed : %f s , Prev step time: -1 s \\ Generating motor dynamics ..."%(t1-t0))

kt=symbols('kt',real=True,positive=True)

omega_1,omega_2,omega_3,omega_4,omega_5,omega_6=symbols('omega_1,omega_2,omega_3,omega_4,omega_5,omega_6',real=True,positive=True)
omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6=symbols('omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6',real=True,positive=True)

omegas_c=Matrix([omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6])
omegas=Matrix([omega_1,omega_2,omega_3,omega_4,omega_5,omega_6])

omegas=omegas+dt*kt*(omegas_c-omegas) if model_motor_dynamics else omegas_c

t2=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Solving blade model ..."%(t2-t0,t2-t1))

"blade dynamics"

b1=symbols('b1',real=True,positive=True)

rho,A,omega,r=symbols('rho,A,omega,r',real=True,positive=True)

c1,c2,c3=symbols('c1,c2,c3',real=True)


ch1,ch2=symbols('ch1,ch2',real=True)


vi=symbols('eta',reals=True)

v2=symbols('v2')
v3=symbols('v3')

if vanilla_force_model:
    T_sum=-c1_s*sum([ omegas[i]**2 for i in range(6)])*R@k_vect
    H_sum=0*k_vect
else:

    T_BET=rho*A*r*omega*(c1*r*omega-c2*(vi-v3)) if not with_ct3 else rho*A*r*omega*(c1*omega*r-c2*(vi-v3)+c3*v2**2)
    
    if structural_relation_idc1:
        T_BET=T_BET.subs(c2, b1*c1-2/b1)
    
    if structural_relation_idc2:
        T_BET=T_BET.subs(c1, c2/b1+2/b1*b1)
        
    T_MOMENTUM_simp=2*rho*A*vi*((vi-v3)+v2) if approx_x_plus_y else 2*rho*A*vi*(vi-v3)
    eq_MOMENTUM_simp=T_BET-T_MOMENTUM_simp
    
    eta=simplify(Matrix([solve(eq_MOMENTUM_simp,vi)[1]])).subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
    
    etas=Matrix([eta.subs(omega,omegas[i]) for i in range(6)])
    
    def et(expr):
        return expr.subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
    
    T_sum=-simplify(sum([et(T_BET).subs(omega,omegas[i]).subs(vi,etas[i]) for i in range(6)]))*R@k_vect
    
    H_tmp=simplify(sum([r*omegas[i]*ch1+ch2*(etas[i]-va_body[2,0]) for i in range(6)]))
    H_sum=-rho*A*H_tmp*(va_NED-va_NED.dot(R@k_vect)*R@k_vect)
    # print(H_sum)
t3=time.time()
"liftdrag forces"
print("Elapsed : %f s , Prev step time: %f s \\ Solving lifrdrag model ..."%(t3-t0,t3-t2))

di,dj,dk=symbols('di,dj,dk',real=True,positive=True)

D=diag(di,di,dk) if di_equal_dj else diag(di,dj,dk)
Fa=-simplify(rho*A*va_NED.norm()*R@D*R.T@va_NED)


t35=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Solving Dynamics ..."%(t35-t0,t35-t3))

g=9.81
new_acc=simplify(g*k_vect+T_sum/m+H_sum/m+Fa/m)
new_v=v+dt*new_acc

t37=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Generating costs ..."%(t37-t0,t37-t35))

alog_i,alog_j,alog_k=symbols("alog_i,alog_j,alog_k",real=True)
alog=Matrix([[alog_i],[alog_j],[alog_k]])

vnext_i,vnext_j,vnext_k=symbols("vnext_i,vnext_j,vnext_k",real=True)
vnext_log=Matrix([[vnext_i],[vnext_j],[vnext_k]])

" constructing opti variables "
" WARNING SUPER WARNING : the order must be the same as in id_variables !!!"

id_variables_sym=[]

id_variables_sym.append(m) if id_mass else None 

id_variables_sym.append(A) if id_blade_geom_coeffs else None
id_variables_sym.append(r) if id_blade_geom_coeffs else None

id_variables_sym.append(c1) if id_blade_coeffs*(not structural_relation_idc2) or vanilla_force_model else None
id_variables_sym.append(c2) if id_blade_coeffs*(not structural_relation_idc1) and (not vanilla_force_model) else None
id_variables_sym.append(c3) if id_blade_coeffs*id_c3 else None
id_variables_sym.append(ch1) if id_blade_coeffs*(not vanilla_force_model) else None
id_variables_sym.append(ch2) if id_blade_coeffs*(not vanilla_force_model) else None


id_variables_sym.append(di) if id_body_liftdrag_coeffs else None
id_variables_sym.append(dj) if id_body_liftdrag_coeffs*(not di_equal_dj) else None
id_variables_sym.append(dk) if id_body_liftdrag_coeffs else None

id_variables_sym.append(vw_i) if id_wind else None
id_variables_sym.append(vw_j) if id_wind else None
    
id_variables_sym.append(kt) if id_time_const else None


id_variables_sym=Matrix(id_variables_sym)

print("\n ID VARIABLES:")
print(id_variables_sym,"\n")

t4=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Gathering identification parameters  ..."%(t4-t0,t4-t37))

t5=time.time()

obs_=new_v if fit_on_v else new_acc

obs_jac_=obs_func.jacobian(id_variables_sym)




t6=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Lambdification ..."%(t6-t0,t6-t5))


X=(m,A,r,rho,
b1,
c1,c2,c3,
ch1,ch2,
di,dj,dk,
vw_i,vw_j,
kt,
dt,
vlog_i,vlog_j,vlog_k,
vpred_i,vpred_j,vpred_k,
alog_i,alog_j,alog_k,
vnext_i,vnext_j,vnext_k,r1,r2,r3,r4,r5,r6,r7,r8,r9,
omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6,
m_scale,A_scale,r_scale,c1_scale,c2_scale,c3_scale,
ch1_scale,ch2_scale,di_scale,dj_scale,dk_scale,
vw_i_scale,vw_j_scale,kt_scale)


# Y=Matrix([new_acc,new_v,omegas,sqerr_a,sqerr_v,Ja.T,Jv.T])
# model_func=lambdify(X,Y, modules='numpy')

obs_func_= lambdify(X,obs_, modules='cupy')
jac_obs_func_ =  lambdify(X,obs_jac_, modules='cupy')


t7=time.time()
print("Elapsed : %f s , Prev step time: %f s \\ Done ..."%(t7-t0,t7-t6))

"cleansing memory"
del(dt,m,
vlog_i,vlog_j,vlog_k,
alog_i,alog_j,alog_k,
vnext_i,vnext_j,vnext_k,
vw_i,vw_j,
kt,
b1,
c1,c2,c3,
ch1,ch2,
di,dj,dk,
rho,A,r,r1,r2,r3,r4,r5,r6,r7,r8,r9,R,
omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6)

# %%   ####### Identification Data Struct
 

non_id_variables={"m":mass,
                  "A":Area,
                  "r":r0,
                  "rho":rho0,
                  "b1":b10,
                  "c1":c10,
                  "c2":c20,
                  "c3":c30,
                  "ch1":ch10,
                  "ch2":ch20,
                  "di":di0,
                  "dj":dj0,
                  "dk":dk0,
                  "vw_i":vwi0,
                  "vw_j":vwj0,
                  "kt":kt0}
id_variables={}

if id_mass:
    id_variables['m']=mass  
    
if id_blade_coeffs:
    if vanilla_force_model:
        id_variables['c1']=c10
    if not structural_relation_idc2:
        id_variables['c1']=c10
    if not structural_relation_idc1 and not vanilla_force_model:
        id_variables['c2']=c20
    if id_c3:
        id_variables['c3']=c30
    if not vanilla_force_model:
        id_variables['ch1']=ch10
        id_variables['ch2']=ch20
  
if id_blade_geom_coeffs:
    id_variables['A']=Area
    id_variables['r']=r0

if id_body_liftdrag_coeffs:
    id_variables['di']=di0
    if not di_equal_dj:
        id_variables['dj']=dj0
    id_variables['dk']=dk0
    
if id_wind:
    id_variables['vw_i']=vwi0
    id_variables['vw_j']=vwj0
    
if id_time_const:
    id_variables['kt']=kt0

"cleaning non_id_variables to avoid having variables in both dicts"
rem=[]
for i in non_id_variables.keys():
    if i in id_variables.keys():
        rem.append(i)
        
for j in rem:
    del(non_id_variables[j])

for i in id_variables.keys():
    id_variables[i]=id_variables[i]/scalers[i]

print("\n ID VARIABLES:")
print(id_variables,"\n")


print("\n NON ID VARIABLES:")
print(non_id_variables,"\n")
    
# %%   ####### IMPORT DATA 
print("LOADING DATA...")
import pandas as pd

log_path="./logs/vol1_ext_alcore/log_real_processed.csv"
# log_path="./logs/vol12/log_real.csv"

raw_data=pd.read_csv(log_path)

print("PROCESSING DATA...")


prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])

if "vol1_ext_alcore" in log_path:
    tmin,tmax=(41,265) 
if "vol2_ext_alcore" in log_path:
    tmin,tmax=(10,140) 
if "vol12" in log_path:
    tmin,tmax=(-1,1e10) 
    
prep_data=prep_data[prep_data['t']>tmin]
prep_data=prep_data[prep_data['t']<tmax]
prep_data=prep_data.reset_index()

for i in range(3):
    prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
    
    
prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
prep_data['t']-=prep_data['t'][0]
prep_data=prep_data.drop(index=[0,len(prep_data)-1])
prep_data=prep_data.reset_index()

data_prepared=prep_data



for i in range(6):
    data_prepared['omega_c[%i]'%(i+1)]=(data_prepared['PWM_motor[%i]'%(i+1)]-pwmmin)/(pwmmax-pwmmin)*U_batt*kv_motor*2*np.pi/60
    
    
print("DATA PROCESS DONE")
    
    
# %% Run funcs
from numba import jit

@jit
def X_to_dict(X,keys_=id_variables.keys()):
    out_dict={}
    for i,key in enumerate(keys_):
        out_dict[key]=X[i]
    return out_dict

@jit
def dict_to_X(input_dict):
    return np.array([input_dict[key] for key in input_dict])

import transforms3d as tf3d
@jit
def arg_wrapping(batch,id_variables,data_index):
    
    i=data_index


    dt=min(batch['dt'][i],1e-2)
    m=mass
    vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]

    alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]

    
    m=non_id_variables['m'] if 'm' in non_id_variables else id_variables['m']
    vw_i=non_id_variables['vw_i'] if 'vw_i' in non_id_variables else id_variables['vw_i']
    vw_j=non_id_variables['vw_j'] if 'vw_j' in non_id_variables else id_variables['vw_j']
    kt=non_id_variables['kt'] if 'kt' in non_id_variables else id_variables['kt']
    b1=non_id_variables['b1'] if 'b1' in non_id_variables else id_variables['b1']
    c1=non_id_variables['c1'] if 'c1' in non_id_variables else id_variables['c1']
    c2=non_id_variables['c2'] if 'c2' in non_id_variables else id_variables['c2']
    c3=non_id_variables['c3'] if 'c3' in non_id_variables else id_variables['c3']
    ch1=non_id_variables['ch1'] if 'ch1' in non_id_variables else id_variables['ch1']
    ch2=non_id_variables['ch2'] if 'ch2' in non_id_variables else id_variables['ch2']
    di=non_id_variables['di'] if 'di' in non_id_variables else id_variables['di']
    dj=non_id_variables['dj'] if 'dj' in non_id_variables else id_variables['dj']
    dk=non_id_variables['dk'] if 'dk' in non_id_variables else id_variables['dk']
    rho=non_id_variables['rho'] if 'rho' in non_id_variables else id_variables['rho']
    A=non_id_variables['A'] if 'A' in non_id_variables else id_variables['A']
    r=non_id_variables['r'] if 'r' in non_id_variables else id_variables['r']
    
    R=tf3d.quaternions.quat2mat(np.array([batch['q[%i]'%(j)][i] for j in range(4)]))
    
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6=np.array([batch['omega_c[%i]'%(j)][i] for j in range(1,7,1)])

            

    
    X=(m,A,r,rho,
    b1,
    c1,c2,c3,
    ch1,ch2,
    di,dj,dk,
    vw_i,vw_j,
    kt,
    dt,
    vlog_i,vlog_j,vlog_k,
    alog_i,alog_j,alog_k,
    *R.flatten(),
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6)
    
    return X
    
@jit
def ekf_run_step(batch,id_variables,P_prev,data_index,Q=Q0,R=R0):
    
    X_arg=arg_wrapping(batch,id_variables,data_index)
    x=dict_to_X().reshape((-1,1))
    P=P_prev
    
    x_pred=x
    P_pred=P+Q
    
    H=jac_obs_func_(X_arg)
    
    vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
    alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
    
    
    mes= np.array((vlog_i,vlog_j,vlog_k) if fit_on_v else (alog_i,alog_j,alog_k)).reshape((-1,1))
    y_pred_ = obs_func_(X_arg)
    
    innovation = mes - y_pred_
    innovation_cov_ = H @ P_pred @ (H).T + R
    innovation_cov_inverse_ = np.linalg.inv(innovation_cov_)
    kalman_gain_ = P_pred @ (H.T) @ innovation_cov_inverse_
    
    x_pred_update_ = x_pred + kalman_gain_ @ innovation
    P_pred_update_ = (np.eye(P_pred.shape[0])-kalman_gain_@H)@P_pred
    
    return x_pred_update_ , P_pred_update_ , y_pred_


def run(data,id_var,P=P0,Q=Q0,R=R0):
    
    current_dict=id_var
    
    x_=np.zeros((len(data),len(id_var)))
    P_=np.zeros((len(data),len(id_var),len(id_var)))
    y_=np.zeros((len(data),3))

    x_[0]=dict_to_X(id_var)
    P_[0]=P0
    
    for j,i in enumerate(batch.index):
        if j>1:
            current_dict=X_to_dict(x_[j-1])
            P_prev=P_[j-1]
            data_index=i
            x_tmp,P_tmp, y_tmp = ekf_run_step(data,current_dict,P_prev,data_index,Q=Q0_,R=R0)
            x_[j],P_[j],y_[j] = x_tmp,P_tmp, y_tmp
            current_dict=X_to_dict(x_tmp)
    return x_,P_,y_

def save_as_json_and_df():
