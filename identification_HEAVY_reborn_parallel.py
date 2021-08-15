

import  numpy as np
# %%   ####### PARAMETERS
import sys
import time
from sympy import *

def main_func(x):
    # generic booleans
    fit_arg,blr,ns=x[0],x[1],x[2]
    
    
    # generic booleans
    
    
    "regresion on a"
    fit_on_v=fit_arg
    used_logged_v_in_model=not fit_arg
    base_lr=1.0 if  blr=="scipy" else blr

    fit_strategy="scipy" if  blr=="scipy" else "custom_gradient"
    nsecs=ns
    model_motor_dynamics=True

    n_epochs=20
    
    log_name="postcrashVRAILOG_fit_v_%s_lr_%s_ns_%s"%(str(fit_on_v),str(base_lr) if blr!="scipy" else 'scipy',str(ns))


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
    
    train_proportion=0.8
    grad_autoscale=False
    
    
    # Log path
    
    # log_path="/logs/vol1_ext_alcore/log_real.csv"
    # log_path="/logs/vol1_ext_alcore/log_real.csv"
    log_path="./logs/vol12/log_real_processed.csv"
    
    
    
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
    
    c10=0.0139055
    c20=0.0386786
    c30=0.0
    ch10=0.0282942
    ch20=0.1
    di0=0.806527
    dj0=0.632052
    dk0=1.59086
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
    
    scalers={}
    scalers['m']=1.0
    scalers['A']=1.0
    scalers['r']=1.0
    scalers['c1']=1.0e-2
    scalers['c2']=1.0e-1
    scalers['c3']=1.0
    scalers['ch1']=5.0e-1
    scalers['ch2']=5.0e-1
    scalers['di']=5.0e1
    scalers['dj']=5.0e1
    scalers['dk']=5.0e1
    scalers['vw_i']=1.0e1
    scalers['vw_j']=1.0e1
    scalers['kt']=1.0e2
    
    
    
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
    
    
    
    # %%   ####### SYMPY PROBLEM 
    
    import time
    
    t0=time.time()
    
    
    print("\nElapsed : %f s , Prev step time: -1 s \\ Generating first symbols ..."%(time.time()-t0))
    
    dt=symbols('dt',positive=True,real=True)
    m=symbols('m',reals=True,positive=True)
    
    m_scale=symbols('m_scale',real=True,positive=True)
    vw_i_scale,vw_j_scale=symbols('vw_i_scale,vw_j_scale',real=True,positive=True)
    kt_scale=symbols('kt_scale',real=True,positive=True)
    A_scale,r_scale=symbols('A_scale,r_scale',real=True,positive=True)
    c1_scale,c2_scale,c3_scale=symbols('c1_scale,c2_scale,c3_scale',real=True,positive=True)
    ch1_scale,ch2_scale=symbols('ch1_scale,ch2_scale',real=True,positive=True)
    di_scale,dj_scale,dk_scale=symbols('di_scale,dj_scale,dk_scale',real=True,positive=True)
    
    m_s=m*m_scale
    
    r1,r2,r3,r4,r5,r6,r7,r8,r9=symbols("r1,r2,r3,r4,r5,r6,r7,r8,r9",real=True)
    R=Matrix([[r1,r2,r3],
              [r4,r5,r6],
              [r7,r8,r9]])
    
    
    vlog_i,vlog_j,vlog_k=symbols("vlog_i,vlog_j,vlog_k",real=True)
    vpred_i,vpred_j,vpred_k=symbols("vpred_i,vpred_j,vpred_k",real=True)
    
    v_i,v_j,v_k=(vlog_i,vlog_j,vlog_k) if used_logged_v_in_model else (vpred_i,vpred_j,vpred_k)
    
    
    vw_i,vw_j=symbols('vw_i,vw_j',real=True)
    vw_i_s,vw_j_s=vw_i*vw_i_scale,vw_j*vw_j_scale
    
    v=Matrix([[v_i],
             [v_j],
             [v_k]])
    
    if not assume_nul_wind:
        va_NED=Matrix([[v_i-vw_i_s],
                        [v_j-vw_j_s],
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
    kt_s=kt*kt_scale
    omega_1,omega_2,omega_3,omega_4,omega_5,omega_6=symbols('omega_1,omega_2,omega_3,omega_4,omega_5,omega_6',real=True,positive=True)
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6=symbols('omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6',real=True,positive=True)
    
    omegas_c=Matrix([omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6])
    omegas=Matrix([omega_1,omega_2,omega_3,omega_4,omega_5,omega_6])
    
    omegas=omegas+dt*kt_s*(omegas_c-omegas) if model_motor_dynamics else omegas_c
    
    t2=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Solving blade model ..."%(t2-t0,t2-t1))
    
    "blade dynamics"
    
    b1=symbols('b1',real=True,positive=True)
    
    rho,A,omega,r=symbols('rho,A,omega,r',real=True,positive=True)
    A_s,r_s=A*A_scale,r*r_scale
    c1,c2,c3=symbols('c1,c2,c3',real=True)
    c1_s,c2_s,c3_s=c1*c1_scale,c2*c2_scale,c3*c3_scale
    
    ch1,ch2=symbols('ch1,ch2',real=True)
    ch1_s,ch2_s=ch1*ch1_scale,ch2*ch2_scale
    
    vi=symbols('eta',reals=True)
    
    v2=symbols('v2')
    v3=symbols('v3')
    
    if vanilla_force_model:
        T_sum=-c1_s*sum([ omegas[i]**2 for i in range(6)])*R@k_vect
        H_sum=0*k_vect
    else:
    
        T_BET=rho*A_s*r_s*omega*(c1_s*r_s*omega-c2_s*(vi-v3)) if not with_ct3 else rho*A_s*r_s*omega*(c1_s*omega*r_s-c2_s*(vi-v3)+c3_s*v2**2)
        
        if structural_relation_idc1:
            T_BET=T_BET.subs(c2, b1*c1-2/b1)
        
        if structural_relation_idc2:
            T_BET=T_BET.subs(c1, c2/b1+2/b1*b1)
            
        T_MOMENTUM_simp=2*rho*A_s*vi*((vi-v3)+v2) if approx_x_plus_y else 2*rho*A_s*vi*(vi-v3)
        eq_MOMENTUM_simp=T_BET-T_MOMENTUM_simp
        
        eta=simplify(Matrix([solve(eq_MOMENTUM_simp,vi)[1]])).subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
        
        etas=Matrix([eta.subs(omega,omegas[i]) for i in range(6)])
        
        def et(expr):
            return expr.subs(v3,va_body[2,0]).subs(v2,sqrt(va_body[0,0]**2+va_body[1,0]**2))
        
        T_sum=-simplify(sum([et(T_BET).subs(omega,omegas[i]).subs(vi,etas[i]) for i in range(6)]))*R@k_vect
        
        H_tmp=simplify(sum([r_s*omegas[i]*ch1_s+ch2_s*(etas[i]-va_body[2,0]) for i in range(6)]))
        H_sum=-rho*A_s*H_tmp*(va_NED-va_NED.dot(R@k_vect)*R@k_vect)
        # print(H_sum)
    t3=time.time()
    "liftdrag forces"
    print("Elapsed : %f s , Prev step time: %f s \\ Solving lifrdrag model ..."%(t3-t0,t3-t2))
    
    di,dj,dk=symbols('di,dj,dk',real=True,positive=True)
    di_s,dj_s,dk_s=di*di_scale,dj*dj_scale,dk*dk_scale
    D=diag(di_s,di_s,dk_s) if di_equal_dj else diag(di_s,dj_s,dk_s)
    Fa=-simplify(rho*A_s*va_NED.norm()*R@D*R.T@va_NED)
    
    
    t35=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Solving Dynamics ..."%(t35-t0,t35-t3))
    
    g=9.81
    new_acc=simplify(g*k_vect+T_sum/m_s+H_sum/m+Fa/m_s)
    new_v=v+dt*new_acc
    
    t37=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Generating costs ..."%(t37-t0,t37-t35))
    
    alog_i,alog_j,alog_k=symbols("alog_i,alog_j,alog_k",real=True)
    alog=Matrix([[alog_i],[alog_j],[alog_k]])
    
    vnext_i,vnext_j,vnext_k=symbols("vnext_i,vnext_j,vnext_k",real=True)
    vnext_log=Matrix([[vnext_i],[vnext_j],[vnext_k]])
    
    
    err_a=Matrix(alog-new_acc)
    err_v=Matrix(vnext_log-new_v)
    
    cost_scaler_a=symbols('C_sa',real=True,positive=True)
    cost_scaler_v=symbols('C_sv',real=True,positive=True)
    
    sqerr_a=Matrix([1.0/cost_scaler_a*(err_a[0,0]**2+err_a[1,0]**2+err_a[2,0]**2)])
    sqerr_v=Matrix([1.0/cost_scaler_v*(err_v[0,0]**2+err_v[1,0]**2+err_v[2,0]**2)])
    
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
    
    Ja=sqerr_a.jacobian(id_variables_sym)
    Jv=sqerr_v.jacobian(id_variables_sym)
    
    t6=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Lambdification ..."%(t6-t0,t6-t5))
    
    
    
    X=(m,A,r,rho,
    b1,
    c1,c2,c3,
    ch1,ch2,
    di,dj,dk,
    vw_i,vw_j,
    kt,
    dt,cost_scaler_a,cost_scaler_v,
    vlog_i,vlog_j,vlog_k,
    vpred_i,vpred_j,vpred_k,
    alog_i,alog_j,alog_k,
    vnext_i,vnext_j,vnext_k,r1,r2,r3,r4,r5,r6,r7,r8,r9,
    omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
    omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6,
    m_scale,A_scale,r_scale,c1_scale,c2_scale,c3_scale,
    ch1_scale,ch2_scale,di_scale,dj_scale,dk_scale,
    vw_i_scale,vw_j_scale,kt_scale)
    
    
    
    Y=Matrix([new_acc,new_v,omegas,sqerr_a,sqerr_v,Ja.T,Jv.T])
    model_func=lambdify(X,Y, modules='numpy')
    
    t7=time.time()
    print("Elapsed : %f s , Prev step time: %f s \\ Done ..."%(t7-t0,t7-t6))
    
    "cleansing memory"
    del(dt,m,cost_scaler_a,cost_scaler_v,
    vlog_i,vlog_j,vlog_k,
    vpred_i,vpred_j,vpred_k,
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
    
    
    "test "
    
    
    
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
        
    "splitting the dataset into nsecs sec minibatches"
    print("SPLIT DATA...")
    
    if nsecs=='all':
        data_batches=[data_prepared]
        N_train_batches=1
        N_val_batches=0
        
    else:
        N_minibatches=round(data_prepared["t"].max()/nsecs) if nsecs >0 else  len(data_prepared)# 22 for flight 1, ?? for flight 2
        N_minibatches=N_minibatches if nsecs!='all' else 1
        
        "if you don't want to minibatch"
        # N_minibatches=len(data_prepared)
        data_batches=[i.drop(columns=[j for j in data_prepared.keys() if (("level" in j ) or ("index") in j) ]) for i in np.array_split(data_prepared, N_minibatches)]
        # print(data_batches)
        data_batches=[i.reset_index() for i in data_batches]
        
        N_train_batches=round(train_proportion*N_minibatches)
        N_val_batches=N_minibatches-N_train_batches
    print("DATA PROCESS DONE")
    
    # %%   ####### MODEL function
    import transforms3d as tf3d 
    
    
    def arg_wrapping(batch,id_variables,scalers,data_index,speed_pred_previous,omegas_pred):
        i=data_index
        
        cost_scaler_v=1.0
        cost_scaler_a=1.0
    
        dt=min(batch['dt'][i],1e-2)
        m=mass
        vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        vpred_i,vpred_j,vpred_k=speed_pred_previous 
        alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
        vnext_i,vnext_j,vnext_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        
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
        omega_1,omega_2,omega_3,omega_4,omega_5,omega_6=omegas_pred
                
        m_scale=scalers['m']
        A_scale=scalers['A']
        r_scale=scalers['r']
        c1_scale,c2_scale,c3_scale=scalers['c1'],scalers['c2'],scalers['c3']
        ch1_scale,ch2_scale=scalers['ch1'],scalers['ch2']
        di_scale,dj_scale,dk_scale=scalers['di'],scalers['dj'],scalers['dk']
        vw_i_scale,vw_j_scale=scalers['vw_i'],scalers['vw_j']
        kt_scale=scalers['kt']
        
        X=(m,A,r,rho,
        b1,
        c1,c2,c3,
        ch1,ch2,
        di,dj,dk,
        vw_i,vw_j,
        kt,
        dt,cost_scaler_a,cost_scaler_v,
        vlog_i,vlog_j,vlog_k,
        vpred_i,vpred_j,vpred_k,
        alog_i,alog_j,alog_k,
        vnext_i,vnext_j,vnext_k,*R.flatten(),
        omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
        omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6,
        m_scale,A_scale,r_scale,c1_scale,c2_scale,c3_scale,
        ch1_scale,ch2_scale,di_scale,dj_scale,dk_scale,
        vw_i_scale,vw_j_scale,kt_scale)
        
        return X
    
    
    def pred_on_batch(batch,id_variables,scalers):
    
        acc_pred=np.zeros((len(batch),3))
        speed_pred=np.zeros((len(batch),3))
        omegas=np.zeros((len(batch),6))    
            
        square_error_a=np.zeros((len(batch),1))    
        square_error_v=np.zeros((len(batch),1))    
        jac_error_a=np.zeros((len(batch),len(id_variables)))
        jac_error_v=np.zeros((len(batch),len(id_variables)))
        
        for i in batch.index:
            print("\r Pred on batch %i / %i "%(i,max(batch.index)), end='', flush=True)
    
            speed_pred_prev=speed_pred[i-1] if i>min(batch.index) else (batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i])
            omegas_pred=omegas[i-1] if i>0 else np.array([batch['omega_c[%i]'%(j)][i] for j in range(1,7,1)])
            X=arg_wrapping(batch,id_variables,scalers,i,speed_pred_prev,omegas_pred)
    
            Y=model_func(*X)
    
            acc_pred[i]=Y[:3].reshape(3,)
            speed_pred[i]=Y[3:6].reshape(3,)
            omegas[i]=Y[6:12].reshape(6,)
            square_error_a[i]=Y[12:13].reshape(1,)
            square_error_v[i]=Y[13:14].reshape(1,)
            jac_error_a[i]=Y[14:14+len(id_variables)].reshape(len(id_variables),)
            jac_error_v[i]=Y[14+len(id_variables):14+2*len(id_variables)].reshape(len(id_variables),)
            
    
        # print("TOTAL TIME:",time.time()-t0)
        return acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v
    
    
    
    # %%   Gradient
    import copy 
    import random
    
    def propagate_gradient(jac_array,id_variables,
                           lr=base_lr):
        
        used_jac=np.mean(jac_array,axis=0)
        print(" --- Propagate %s grad : "%(log_name))
        
        datatoplot=np.c_[used_jac,[id_variables[k]*scalers[k] for k in id_variables.keys()]].T
        toplot=pd.DataFrame(data=datatoplot,columns=id_variables.keys(),index=['J','value'])
        
    
        print(toplot.transpose())
        
        new_dic=copy.deepcopy(id_variables)
        
        for i,key in enumerate(id_variables):
            new_dic[key]=np.clip(id_variables[key]-lr*used_jac[i],bounds[key][0],bounds[key][1])
            
        return new_dic
    
    def X_to_dict(X,keys_=id_variables.keys()):
        out_dict={}
        for i,key in enumerate(keys_):
            out_dict[key]=X[i]
        return out_dict
    
    def dict_to_X(input_dict):
        return np.array([input_dict[key] for key in input_dict])
    
    
    def fun_cost_scipy(X,batch,scalers,writtargs):
        
        "X is the dict_to_X of id_variables"
        "dict reconstruction "

        id_var=X_to_dict(X)


        acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch,id_var,scalers)
        
        used_jac=jac_error_v if fit_on_v else jac_error_a
        used_err=square_error_v if fit_on_v else square_error_a
        J=np.mean(used_jac,axis=0)
        C=np.mean(used_err,axis=0)
        
        print(len(id_var),J.shape)
        datatoplot=np.c_[J,[id_var[k]*scalers[k] for k in id_var.keys()]].T
        toplot=pd.DataFrame(data=datatoplot,columns=id_var.keys(),index=['J','value'])
        print("\n %s------ Cost (in scipy minim): %f\n"%(log_name,C))
        print(toplot.transpose())
        
        n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,t0=writtargs

        realvals={}
        for i in id_var.keys():
            realvals[i]=id_var[i]*scalers[i]
            

        saver(name="epoch_%i_batch_%i_t_%f"%(n,k,time.time()-t0),save_path=spath,
        id_variables=realvals,
        train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
        train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
        val_sc_a=val_sc_a,
        val_sc_v=val_sc_v,
        total_sc_a=total_sc_a,
        total_sc_v=total_sc_v)
        
        return C,J
    
    # %% Train loop
    from scipy.optimize import minimize
    
    def train_loop(data_batches,id_var,n_epochs=n_epochs):
    
        id_variables=copy.deepcopy(id_var)
        print("Copy ...")
        temp_shuffled_batches=copy.deepcopy(data_batches)
        print("Done")
        print("INIT")
    
        # acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared,id_variables)
        # print("Inference on full dataset takes : %i"%(round(time.time()-ti0)))
        
        # total_sc_a=np.sqrt(np.mean(square_error_a,axis=0))
        # total_sc_v=np.sqrt(np.mean(square_error_v ,axis=0))
        total_sc_a=-1
        total_sc_v=-1
        print('\n###################################')
        print('############# Begin ident ###########')
        print("id_variables=",id_variables,
              "train_sc_a=",-1,
              "train_sc_v=",-1,
              "val_sc_a=",-1,
              "val_sc_v=",-1,
              "total_sc_a=",total_sc_a,
              "total_sc_v=",total_sc_v,)
        print('###################################\n')
    
        realvals={}
        for i in id_variables.keys():
            realvals[i]=id_variables[i]*scalers[i]
    
        saver(name="start_",save_path=spath,
              id_variables=realvals,
              train_sc_a=-1,
              train_sc_v=-1,
              val_sc_a=-1,
              val_sc_v=-1,
              total_sc_a=total_sc_a,
              total_sc_v=total_sc_v)
        
        print("Entering training loop...")
        for n in range(n_epochs):
            "begin epoch"
            
            train_sc_a,train_sc_v=0,0
            val_sc_a,val_sc_v=0,0
            total_sc_a,total_sc_v=0,0
            
            random.shuffle(temp_shuffled_batches)
            
            
            for k,batch_ in enumerate(temp_shuffled_batches[:N_train_batches]):
                if fit_strategy not in ('custom_gradient','scipy'):
                    print(" ERROR WRONG FIT STRATEGY !!!!")
                    break
                                        
                if fit_strategy=="custom_gradient":
                
                    acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables,scalers)
                    id_variables=propagate_gradient(jac_error_v if fit_on_v else jac_error_a,id_variables)
                    
                    train_sc_a+=np.sqrt(np.mean(square_error_a,axis=0))
                    train_sc_v+=np.sqrt(np.mean(square_error_v,axis=0))
                    print(" %s --- EPOCH : %i/%i || Train batch : %i/%i || PROGRESS: %i/%i [ta,tv]=[%f,%f]"%(log_name,n,
                                                                                                      n_epochs,
                                                                                                      k,N_train_batches,
                                                                                                      k,len(temp_shuffled_batches),
                                                                                                      np.sqrt(np.mean(square_error_a,axis=0)),
                                                                                                      np.sqrt(np.mean(square_error_v,axis=0))))
                    
                    realvals={}
                    for i in id_variables.keys():
                        realvals[i]=id_variables[i]*scalers[i]

                    if ns!=-1 or k//100==0 :

                        saver(name="epoch_%i_batch_%i"%(n,k),save_path=spath,
                          id_variables=realvals,
                          train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                          train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                          val_sc_a=val_sc_a,
                          val_sc_v=val_sc_v,
                          total_sc_a=total_sc_a,
                          total_sc_v=total_sc_v)
                    
                elif fit_strategy=="scipy":
                    
                    X_start=dict_to_X(id_variables)
                    bnds=[bounds[i] for i in id_variables]
                    
                    writtargs=[n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,time.time()]
                       
                    sol_scipy=minimize(fun_cost_scipy,
                                       X_start,
                                       args=(batch_,scalers,writtargs),
                                       bounds=bnds,
                                        jac=True)#,options={"maxiter":1})
                    
                    id_variables=X_to_dict(sol_scipy["x"])
                    current_score=sol_scipy["fun"]
                    current_score_label="tv" if fit_on_v else "ta"
                    print(" %s --- EPOCH : %i/%i || Train batch : %i/%i || PROGRESS: %i/%i [%s]=[%f]"%(log_name,n,
                                                                                                      n_epochs,
                                                                                                      k,N_train_batches,
                                                                                                      k,len(temp_shuffled_batches),
                                                                                                      current_score_label,
                                                                                                      current_score))  
                    
            train_sc_a/=N_train_batches
            train_sc_v/=N_train_batches
            if ns!="all":
                for k,batch_ in enumerate(temp_shuffled_batches[N_train_batches:]):
        
                    acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables,scalers)
                    
                    val_sc_a+=np.mean(square_error_a,axis=0)
                    val_sc_v+=np.mean(square_error_v ,axis=0)   
                    print(" %s --- EPOCH : %i/%i || Eval batch : %i/%i || PROGRESS: %i/%i [va,vv]=[%f,%f]"%(log_name,
                                                                                                      n,n_epochs,
                                                                                                      k,N_val_batches,
                                                                                                      k+N_train_batches,len(temp_shuffled_batches),
                                                                                                      np.sqrt(np.mean(square_error_a,axis=0)),
                                                                                                      np.sqrt(np.mean(square_error_v,axis=0))))
                    realvals={}
                    for i in id_variables.keys():
                        realvals[i]=id_variables[i]*scalers[i]
                        
                    saver(name="epoch_%i_batch_%i"%(n,k+N_train_batches),save_path=spath,
                      id_variables=realvals,
                      train_sc_a=train_sc_a,
                      train_sc_v=train_sc_v,
                      val_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                      val_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                      total_sc_a=total_sc_a,
                      total_sc_v=total_sc_v)


            
            val_sc_a/=N_val_batches
            val_sc_v/=N_val_batches
    
            acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared,id_variables,scalers)
            total_sc_a=np.sqrt(np.mean(square_error_a,axis=0))
            total_sc_v=np.sqrt(np.mean(square_error_v ,axis=0))
            
            realvals={}
            for i in id_variables.keys():
                realvals[i]=id_variables[i]*scalers[i]
    
            print('\n###################################')
            print('############# END EPOCH ###########')
            print("id_variables=",realvals,
                  "train_sc_a=",train_sc_a,
                  "train_sc_v=",train_sc_v,
                  "val_sc_a=",val_sc_a,
                  "val_sc_v=",val_sc_v,
                  "total_sc_a=",total_sc_a,
                  "total_sc_v=",total_sc_v,)
            print('###################################\n')
    
            saver(name="epoch_%i"%(n),save_path=spath,
                  id_variables=realvals,
                  train_sc_a=train_sc_a,
                  train_sc_v=train_sc_v,
                  val_sc_a=val_sc_a,
                  val_sc_v=val_sc_v,
                  total_sc_a=total_sc_a,
                  total_sc_v=total_sc_v)
            
    train_loop(data_batches,id_variables,n_epochs=n_epochs)
    return 0

from multiprocessing import Pool

if __name__ == '__main__':
    
    # blr_range=['scipy',1e-2]
    # ns_range=['all',25,5,-1]
    # fit_arg_range=[True,False]
    
    blr_range=['scipy']
    ns_range=[15]
    fit_arg_range=[True,False]
    
    rem=[[True, 5e-3, 25],
         [False, 5e-3, 25]]
    
    
    x_r=[[i,j,k] for j in blr_range for i in  fit_arg_range  for k in ns_range ]
    x_r=[i for i in x_r if i not in rem]
    
    x_r=[[True, 'scipy', 15],
         [False, 'scipy', 15],
         [True, 'scipy', 'all'],
         [False, 'scipy', 'all'],
         [True, 1e-6, -1],
         # [False, 1e-6, -1],
         [True, 1e-5, -1],
         # [False, 1e-5, -1],
         # [True, 1e-4, -1],
         # [True, 1e-3, 5],
         [False, 1e-5, 5],
         [True, 'scipy', 25],
         [False, 'scipy', 25]]
    
    print(x_r,len(x_r))

    pool = Pool(processes=len(x_r))
    alidhali=input('LAUNCH ? ... \n >>>>')
    pool.map(main_func, x_r)

