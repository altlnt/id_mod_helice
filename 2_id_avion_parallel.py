
import dill as dill
import  numpy as np
# %%   ####### PARAMETERS
# import sys
import time
# from sympy import Symbol, Matrix, symbols, sin, lambdify
import gc 
# from collections import OrderedDict
import os 
import pandas as pd
from numba import jit 

"cette fonction est le main"
"on utilise le multiprocessing pour tester plusieurs metaparamètres"
"elle est appelée par le process"

def main_func(x):
    
    
    # récupération des arguments
    fit_arg,blr,ns=x[0],x[1],x[2]
    
    bo=False
    # les booleans qui déterminent plusieurs comportements, voir coms     

    fit_on_v=fit_arg #est ce qu'on fit sur la vitesse prédite ou sur l'acc
    used_logged_v_in_model=not fit_arg
    
    # si on utilise l'optimizer scipy, on passe 'scipy' en argument
    # sinon le learning rate de la descente de gradient est base_lr
    
    base_lr=1.0 if  blr=="scipy" else blr
    fit_strategy="scipy" if  blr=="scipy" else "custom_gradient"

    # si wind_signal, on considère la valeur du vent à chaque instant
    # est un paramètre d'optimisation
    # sinon, le vent est considéré comme une constante
    
    wind_signal=False
    assume_nul_wind=True # if true, le signal de vent constant vaut zéro
    nsecs=ns
    # nsecs désigne la taille du batch en secondes
    


    n_epochs=20 #le nombre d'epochs
    
    # log_name désigne le nom des fichiers que l'on enregistrera
    # on customisera chaque fichier avec le numero de l'epoch et la timestamp
    log_name="3_SEPTEMBRE_fit_v_%s_lr_%s_ns_%s"%(str(fit_on_v),str(base_lr) if blr!="scipy" else 'scipy',str(ns))

    #                   CI DESSOUS LES PARAMETRES PROPRES AU MODELE
    


    #                  Ci dessous, on décide quels paramètres on souhaite identifier
    id_mass=False
    id_wind=not assume_nul_wind

    
    
    train_proportion=0.8 #proportion data train vs validation
    
    log_path=os.path.join('./logs/avion/vol1/log_real_processed.csv')     
    save_dir_name="results"

    # Paramètres utilitaires
    
    mass=8.5
    g=np.array([0,0,9.81])
    
    Aire_1,Aire_2,Aire_3,Aire_4,Aire_5 =    0.62*0.262* 1.292 * 0.5,\
                                        0.62*0.262* 1.292 * 0.5, \
                                        0.34*0.1* 1.292 * 0.5,\
                                        0.34*0.1* 1.292 * 0.5, \
                                        1.08*0.31* 1.292 * 0.5
    Aire_list = [Aire_1,Aire_2,Aire_3,Aire_4,Aire_5]
    cp_1,cp_2,cp_3,cp_4,cp_5 = np.array([-0.013,0.475,-0.040],       dtype=float).flatten(), \
                            np.array([-0.013,-0.475,-0.040],      dtype=float).flatten(), \
                            np.array([-1.006,0.17,-0.134],    dtype=float).flatten(),\
                            np.array([-1.006,-0.17,-0.134],   dtype=float).flatten(),\
                            np.array([0.021,0,-0.064],          dtype=float).flatten()
    cp_list=[cp_1,cp_2,cp_3,cp_4,cp_5]
    
    Area=np.pi*(11.0e-02)**2
    r0=11e-02
    rho0=1.204
    kv_motor=800.0
    pwmmin=1075.0
    pwmmax=1950.0
    U_batt=16.8
    
    vwi0=0.0
    vwj0=0.0
    vwk0=0.0
    
    alpha_0=0.07
    alpha_s = 0.3391428111
    delta_s = 15.0*np.pi/180
    cd0sa_0 = 0.9
    cd0fp_0 = 0.010
    cd1sa_0 = 2
    cl1sa_0 = 5 
    cd1fp_0 = 2.5 
    coeff_drag_shift_0= 0.5 
    coeff_lift_shift_0= 0.05 
    coeff_lift_gain_0= 2.5
    C_t = 1.1e-4
    C_q = 1e-8
    C_h = 1e-4

    
    
    physical_params=[mass,
    Area,
    r0,
    rho0,
    kv_motor,
    pwmmin,
    pwmmax,
    U_batt,
    cd0sa_0,
    cd0fp_0,
    cd1sa_0,
    cl1sa_0 ,
    cd1fp_0,
    coeff_drag_shift_0,
    coeff_lift_shift_0,
    coeff_lift_gain_0,
    vwi0,
    vwj0,
    C_t]
    
    
    # Bounds and scaling factors
    bounds={}
    bounds['m']=(0,np.inf)
    bounds['A']=(0,np.inf)
    bounds['r']=(0,np.inf)
    bounds['cd0sa']=(0,np.inf)
    bounds['cd0fp']=(-np.inf,np.inf)
    bounds['cd1sa']=(-np.inf,np.inf)
    bounds['cl1sa']=(-np.inf,np.inf)
    bounds['cd1fp']=(-np.inf,np.inf)
    bounds['coeff_drag_shift']=(0,np.inf)
    bounds['coeff_lift_shift']=(0,np.inf)
    bounds['coeff_lift_gain']=(0,np.inf)
    bounds['vw_i']=(-15,15)
    bounds['vw_j']=(-15,15)
    bounds['Ct']=(0,1)

    "scaler corresponds roughly to the power of ten of the parameter"
    "it does not have to though, it may be used to improve the grad descent"
    
    scalers={}
    for i in bounds:
        scalers[i]=1.0
    
    
    
    metap={
            "used_logged_v_in_model":used_logged_v_in_model,
            "fit_on_v":fit_on_v,
            "wind_signal":wind_signal,
            "assume_nul_wind":assume_nul_wind,
            "log_path":log_path,
            "base_lr":base_lr,
            "n_epochs":n_epochs,
            "nsecs":nsecs,
            "train_proportion":train_proportion,
            "[mass,Area,r0,rho0,kv_motor,pwmmin,pwmmax,U_batt,cd0sa_0, cd0fp_0,cd1sa_0,cl1sa_0 ,cd1fp_0,coeff_drag_shift_0,coeff_lift_shift_0,coeff_lift_gain_0,vwi0vwj0,C_t0]":physical_params,
            "bounds":bounds}
    
    
    
    print(" META PARAMS \n")
    [print(i,":", metap[i]) for i in metap.keys()]
    
    
    
    # %%   ####### Saving function
    
    import json
    import datetime
    
    #generating a new log name if none is provided
    if log_name=="":
        log_name=str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        
    spath=os.path.join(os.getcwd(),save_dir_name,log_name)

    #on enlève les logs avec le même nom
    import shutil
    try:
        shutil.rmtree(spath)
    except:
        pass
    os.makedirs(spath)
    
    #on sauvegarde les paramètres de départ
    with open(os.path.join(spath,'data.json'), 'w') as fp:
        json.dump(metap, fp)
        
    # la fonction saver va être utilisée souvent 
    
    def saver(name=None,save_path=os.path.join(os.getcwd(),"results_tests"),**kwargs):
        dname=str(int(time.time())) if (name is None) else name

        "le fonctionnement est simple:on prend tout ce qu'il y a dans **kwargs"
        "et  on le met dans un dictionnaire qu'on save dans un .json"
        
        D={}
        for i in kwargs:
            if type(kwargs[i])==dict:
                for j in kwargs[i]:
                    D[j]=kwargs[i][j]
            else:
                D[i]=kwargs[i].tolist() if isinstance(kwargs[i],np.ndarray) else kwargs[i]
        
        with open(os.path.join(save_path,'%s.json'%(dname)), 'w') as fp:
            json.dump(D, fp)
        return 0 
    
    
    # %%   ####### SYMPY PROBLEM 
    
    " le but de cette partie est de génerer une fonction qui renvoie : "
    " new_acc,new_v,sqerr_a,sqerr_v,Ja.T,Jv.T    , c'est à dire        "
    " l'acc pred, la vitesse pred, l'erreur quad sur acc, l'erreur quad sur v"
    " la jacobienne de l'erreur quad sur l'acc et la vitesse "
        
    
    def Rotation(R,angle):
        c, s = np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)
        r = np.array([[1,0, 0], [0,c, -s],[0,s, c]] , dtype=np.float)
        return R @ r
    #CI DESSOUS : on spécifie quelles variables sont les variables d'identif
   
    model_func = dill.load(open('./.Funcs/model_func_'+str(used_logged_v_in_model)+"simple_"+str(bo),'rb'))[0]
    function_moteur_physique=  dill.load(open('./.Funcs/model_func_'+str(used_logged_v_in_model)+"simple_"+str(bo),'rb'))[1]
    # CI DESSOUS : on spécifie quelles variables sont les variables d'identif
    
    # t7=time.time()
    
    "cleansing memory"
    
    # très recommandé d'effacer les variables de sympy pour éviter les soucis 
    # dans le restes des fonctions
    
    # del(dt,m,cost_scaler_a,cost_scaler_v,
    # vlog_i,vlog_j,vlog_k,
    # vpred_i,vpred_j,vpred_k,
    # alog_i,alog_j,alog_k,
    # vnext_i,vnext_j,vnext_k,
    # vw_i,vw_j,
    # kt,
    # b1,
    # c1,c2,c3,
    # ch1,ch2,
    # di,dj,dk,
    # rho,A,r,r1,r2,r3,r4,r5,r6,r7,r8,r9,R,
    # omega_1,omega_2,omega_3,omega_4,omega_5,omega_6,
    # omega_c1,omega_c2,omega_c3,omega_c4,omega_c5,omega_c6)
    
    
    "test "
    
    
    
    # %%   ####### IMPORT DATA 
    
    # a la fin de ce bloc, on obtient une liste de datasets, correspondant
    # aux batches. 
    
    print("LOADING DATA...")
    
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
    prep_data=prep_data.reset_index()
    
    
    for i in range(3):
        prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
        
        
    prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
    prep_data['t']-=prep_data['t'][0]
    prep_data=prep_data.drop(index=[0,len(prep_data)-1])
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)//20]
    
    
    
    def scale_to_01(df):

        return (df-df.min())/(df.max()-df.min())
    
    data_prepared['omega_c[5]']=(data_prepared['PWM_motor[5]']-1000)*925.0/1000
    "splitting the dataset into nsecs sec minibatches"
    
    print("SPLIT DATA...")
    
    if nsecs=='all':
        data_batches=[data_prepared]
        N_train_batches=1
        N_val_batches=0
        
    else:
        N_minibatches=round(data_prepared["t"].max()/nsecs) if nsecs >0 else  len(data_prepared)# 22 for flight 1, ?? for flight 2
        N_minibatches=N_minibatches if nsecs!='all' else 1

        data_batches=[i.drop(columns=[j for j in data_prepared.keys() if (("level" in j ) or ("index") in j) ]) for i in np.array_split(data_prepared, N_minibatches)]

        data_batches=[i.reset_index() for i in data_batches]
        
        N_train_batches=round(train_proportion*N_minibatches)
        N_val_batches=N_minibatches-N_train_batches
    print("DATA PROCESS DONE")
            
    # print("Importing model func...")
    # %%   ####### Identification Data Struct
    
    # On répartit les variables entre deux dicts: id_variables et non_id_variables
    # Cette étape est liée, par les booléens utilisés, à la premi_re étape
    
    non_id_variables={"m":mass,
                      "A":Area,
                      "r":r0,
                      "rho":rho0,
                    'cd0sa':cd0sa_0,
                    'cd0fp':cd0fp_0,
                    'cd1sa':cd1sa_0,
                    'cl1sa':cl1sa_0,
                    'cd1fp':cd1fp_0,
                    'coeff_drag_shift':coeff_drag_shift_0,
                    'coeff_lift_shift':coeff_lift_shift_0,
                    'coeff_lift_gain':coeff_lift_gain_0,
                      "vw_i":vwi0,
                      "vw_j":vwj0,
                      "vw_k":vwk0,
                      "alpha_stall":alpha_s,
                      "largeur_stall":delta_s,
                      "Ct": C_t, 
                      "Cq": C_q, 
                      "Ch": C_h
                      }
    id_variables={}
    for key_ in ('cd0sa','cd0fp',
                 'cd1sa','cl1sa','cd1fp',
                 'coeff_drag_shift','coeff_lift_shift',
                 'coeff_lift_gain'):
        id_variables[key_]=non_id_variables[key_]
    
    if id_mass:
        id_variables['m']=mass  
        
        
    if id_wind:
        id_variables['vw_i']=vwi0
        id_variables['vw_j']=vwj0
        if wind_signal:
            id_variables['vw_i']=vwi0*np.zeros(len(data_prepared))
            id_variables['vw_j']=vwj0*np.zeros(len(data_prepared)) 
            
    
    "cleaning non_id_variables to avoid having variables in both dicts"
    rem=[]
    for i in non_id_variables.keys():
        if i in id_variables.keys():
            rem.append(i)
            
    for j in rem:
        del(non_id_variables[j])
    
    for i in id_variables.keys():
        id_variables[i]=id_variables[i]/scalers[i]
    

    # %%   ####### MODEL function
    
    # ici, on définit les fonctions que l'on appellera dans la partie 
    # optimisation. 
    
    
    import transforms3d as tf3d 
    import copy 
    
    def arg_wrapping(batch,id_variables,data_index,speed_pred_previous):
        
        "cette fonction sert à fabriquer, à partir des inputs, l'argument que "
        "l'on enverra en input à la fonction lambdifiée de la partie sympy    "
        
        "batch est un dataframe"
        "id_variables sont les variables d'identification"
        "scalers sont les coefficients de mise à l'échelle"
        "data_index est un entier, qui sert à récupérer la bonne valeur "
        "dans le batch de données"
        "speed_pred_previous est la vitesse prédite précédente"
        "omegas_pred est la vitesse angulaire pred précédente,à partir de kt"
        
        i=data_index
        
        # cost_scaler_v=1.0
        # cost_scaler_a=1.0
        
        dt=min(batch['dt'][i],1e-2)
        
        vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        v_log = np.array([[vlog_i],
                       [vlog_j],
                       [vlog_k]])
        
        vpred_i,vpred_j,vpred_k=speed_pred_previous 
        v_pred=np.array([[vpred_i],
                       [vpred_j],
                       [vpred_k]])
        
        alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
        alog=np.array([[alog_i],
                       [alog_j],
                       [alog_k]])
        
        # vnext_i,vnext_j,vnext_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        
        m=non_id_variables['m'] if 'm' in non_id_variables else id_variables['m']
        
        vw_i=non_id_variables['vw_i'] if 'vw_i' in non_id_variables else id_variables['vw_i']
        vw_j=non_id_variables['vw_j'] if 'vw_j' in non_id_variables else id_variables['vw_j']
        
        if wind_signal:
            vw_i,vw_j=vw_i[i],vw_j[i]
            
        v_W=np.array([[vw_i],
                     [vw_j], 
                     [vwk0]])
        
        Omega=np.zeros(3)
        
        cd0sa=non_id_variables['cd0sa'] if 'cd0sa' in non_id_variables else id_variables['cd0sa']
        cd0fp=non_id_variables['cd0fp'] if 'cd0fp' in non_id_variables else id_variables['cd0fp']
        cd1sa=non_id_variables['cd1sa'] if 'cd1sa' in non_id_variables else id_variables['cd1sa']
        cl1sa=non_id_variables['cl1sa'] if 'cl1sa' in non_id_variables else id_variables['cl1sa']
        cd1fp=non_id_variables['cd1fp'] if 'cd1fp' in non_id_variables else id_variables['cd1fp']
        coeff_drag_shift=non_id_variables['coeff_drag_shift'] if 'coeff_drag_shift' in non_id_variables else id_variables['coeff_drag_shift']
        coeff_lift_shift=non_id_variables['coeff_lift_shift'] if 'coeff_lift_shift' in non_id_variables else id_variables['coeff_lift_shift']
        coeff_lift_gain=non_id_variables['coeff_lift_gain'] if 'coeff_lift_gain' in non_id_variables else id_variables['coeff_lift_gain']
        C_t=non_id_variables['Ct'] if 'Ct' in non_id_variables else id_variables['Ct']

        R=tf3d.quaternions.quat2mat(np.array([batch['q[%i]'%(j)][i] for j in range(4)]))
        R_list =[R,R,Rotation(R,-45),Rotation(R,-135),R]
        
        
        "reverse mixing"
        pwm_null_angle=1527
        RW_delta=batch['PWM_motor[1]'][i]-pwm_null_angle
        LW_delta=batch['PWM_motor[2]'][i]-pwm_null_angle
        RVT_delta=batch['PWM_motor[3]'][i]-pwm_null_angle
        LVT_delta=batch['PWM_motor[4]'][i]-pwm_null_angle
        
        delta0_list=np.array([RW_delta,-LW_delta,RVT_delta,-LVT_delta,0.0])  
        
        delta_pwm=500
        delta0_list/=delta_pwm
        
        delta0_list=delta0_list*15.0/180.0*np.pi
        
        ## Commande entre -15:15 pour les 4 premier terme, le dernier terme vaut 0 (pour l'homogéinité des longueur)
        omega_rotor = batch['omega_c[5]'][i]                    ## Vitesse de rotation des helices (supposé les mêmes pour les deux moteurs)
        alpha_list=[0,0,0,0,0]
        for p, cp in enumerate(cp_list) :          # Cette boucle calcul les coefs aéro pour chaque surface 
            VelinLDPlane   = function_moteur_physique[0](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
            dragDirection  = function_moteur_physique[1](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
            liftDirection  = function_moteur_physique[2](Omega, cp, v_pred.flatten(), v_W.flatten(), R_list[p].flatten())
            alpha_list[p] =  -function_moteur_physique[3](dragDirection, liftDirection, np.array([[1],[0],[0]]).flatten(), VelinLDPlane)
    
        X=(alog.flatten(),v_log.flatten(),dt, Aire_list, Omega.flatten(), R.flatten(), v_pred.flatten(), v_W.flatten(), cp_list, alpha_list, alpha_0, \
           alpha_s, delta0_list.flatten(), delta_s, cl1sa, cd1fp, coeff_drag_shift, coeff_lift_shift, coeff_lift_gain,\
               cd0fp, cd0sa, cd1sa, C_t, C_q, C_h, omega_rotor, g.flatten(), m)

        return X
    

    def pred_on_batch(batch,id_variables):
        
        "si n est la taille du batch"
        "cette fonction sert à appeler n fois la fonction lambdifiée"
        " de sympy "
        
        "on obtient n acc prédites, n vitesses prédites, n jacobiennes...."
        
        
        "batch est un dataframe"
        "id_variables sont les variables d'identification"
        "scalers sont les coefficients de mise à l'échelle"
        
        acc_pred=np.zeros((len(batch),3))
        speed_pred=np.zeros((len(batch),3))
        
            
        square_error_a=np.zeros((len(batch),1))    
        square_error_v=np.zeros((len(batch),1))    
        jac_error_a=np.zeros((len(batch),len(id_variables)))
        jac_error_v=np.zeros((len(batch),len(id_variables)))
        
        for i in batch.index:
        
            print("\r Pred on batch %i / %i "%(i,max(batch.index)), end='', flush=True)
        
            speed_pred_prev=speed_pred[i-1] if i>min(batch.index) else (batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i])
        
            X=arg_wrapping(batch,id_variables,i,speed_pred_prev)
        
            Y=model_func(*X)

            acc_pred[i]=Y[:3].reshape(3,)
            speed_pred[i]=Y[3:6].reshape(3,)
            square_error_a[i]=Y[6:7].reshape(1,)
            square_error_v[i]=Y[7:8].reshape(1,)
            jac_error_a[i]=Y[8:8+len(id_variables)].reshape(len(id_variables),)
            jac_error_v[i]=Y[8+len(id_variables):8+2*len(id_variables)].reshape(len(id_variables),)
            
        
        
        return acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v
    
       
            
    def X_to_dict(X,base_dict=id_variables):
            
            "sert à transformer un vecteur en dictionnaire "
            
            out_dict={}
            index_j=0
            for i,key in enumerate(base_dict.keys()):
                    out_dict[key]=X[index_j:index_j+len(base_dict[key])] if isinstance(base_dict[key],np.ndarray) else X[index_j] 
                    index_j+=len(base_dict[key]) if isinstance(base_dict[key],np.ndarray) else 1
            return out_dict
        
    def dict_to_X(input_dict):
            
            "sert à transformer un dictinonaire en vecteur "
    
            out=np.r_[tuple([np.array(input_dict[key]).flatten() for key in input_dict])]
            return out
        
    def prepare_dict(id_var,index):
            "sert à processer le dictionnaire id_var"
            "si wind_signal, on ne retient que la valeur index de id_var"
            
            if not wind_signal:
                return id_var
            else:
                newdic={}
                for k in id_var:
                    newdic[k]=id_var[k]
                
                newdic['vw_i'],newdic['vw_j']=newdic['vw_i'][index],newdic['vw_j'][index]
                return newdic
    
    def propagate_gradient(jac_array,id_variables,
                               lr=base_lr):
            "à partir de n jacobiennes et du dictionnaire des variables "
            "d'identification, on calcule les nouveaux paramètres"
            "cette fonction est la fonction que l'on utilisera si on choisit"
            "la descente de gradient comme algorithme de minmisation"
    
            X=dict_to_X(id_variables)
    
            print(" --- Propagate %s grad : "%(log_name))
            
    
            J=np.zeros(len(X))
            index_j=0
            for i,key in enumerate(id_variables.keys()):
                if type(id_variables[key]) is not np.ndarray:
    
                    J[index_j]=np.mean(jac_array[:,i])
                    index_j+=1
                else:
                    J[index_j:index_j+len(id_variables[key])]=jac_array[:,i]
                    index_j+=len(id_variables[key])
    
            new_X=X-lr*J
            new_dic=X_to_dict(new_X,id_variables)
            return new_dic
    
    def fun_cost_scipy(X,batch,scalers,writtargs):
            
            "cette fonction est la fonction que l'on utilisera si on choisit"
            "scipy comme algorithme de minmisation"
            "elle renvoie le coût, et la jacobienne"
            
            "X is the dict_to_X of id_variables"
            "dict reconstruction "
            
            n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,t0,write_this_step,id_variables_base=writtargs
    
    
            id_var=X_to_dict(X,base_dict=id_variables_base)
    
            acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch,id_var,scalers)
            
            used_jac=jac_error_v if fit_on_v else jac_error_a
            used_err=square_error_v if fit_on_v else square_error_a
    
            #calcul de la jacobienne
            J=np.zeros(len(X))
            index_j=0
            for i,key in enumerate(id_var.keys()):
                if type(id_var[key]) is not np.ndarray:
    
                    J[index_j]=np.mean(used_jac[:,i])
                    index_j+=1
                else:
                    J[index_j:index_j+len(id_var[key])]=used_jac[:,i]
                    index_j+=len(id_var[key])
    
    
    
            C=np.mean(used_err,axis=0)
            
            print("\n %s------ Cost (in scipy minim): %f\n"%(log_name,C))
            
            "on repasse les valeurs dans leur valeur physique "
            realvals={}
            for i in id_var.keys():
                realvals[i]=id_var[i]*scalers[i] if ('vw' not in i) else id_var[i][0]*scalers[i]
                
            if write_this_step:
                saver(name="epoch_%i_batch_%i_t_%f"%(n,k,time.time()-t0),save_path=spath,
                id_variables=realvals,
                train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                val_sc_a=val_sc_a,
                val_sc_v=val_sc_v,
                total_sc_a=total_sc_a,
                total_sc_v=total_sc_v)
            if wind_signal and write_this_step:
                windsave_df=pd.DataFrame(data=np.array([id_var['vw_i'],id_var['vw_j']]).T,columns=['w_i','w_j'])
                nsave="WINDSIG_epoch_%i_batch_%i_t_%f"%(n,k,time.time())+".csv"
                windsave_df.to_csv(os.path.join(spath,nsave))
                del(windsave_df)
            
            return C,J
            
    # %% Train loop
    from scipy.optimize import minimize
    import random
    def train_loop(data_batches,id_var,n_epochs=n_epochs):
    
        #     on commence par copier le dict courant et randomiser, puis on sauvegarde
    
        id_variables=copy.deepcopy(id_var)
        print("Copy ...")
        temp_shuffled_batches=copy.deepcopy(data_batches)
        print("Done")
        print("INIT")
    
        total_sc_a=-1
        total_sc_v=-1
        print('\n###################################')
        print('############# Begin ident ###########')
        print("id_variables=",prepare_dict(id_variables,0),
              "train_sc_a=",-1,
              "train_sc_v=",-1,
              "val_sc_a=",-1,
              "val_sc_v=",-1,
              "total_sc_a=",total_sc_a,
              "total_sc_v=",total_sc_v,)
        print('###################################\n')
    
    
        "on repasse les valeurs dans leur valeur physique "
        realvals={}
        for i in id_variables.keys():
            realvals[i]=prepare_dict(id_variables,0)[i]*scalers[i]
    
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
            
            "train score correspond au score sur le dataset de d'entrainement"
            "val score correspond au score sur le dataset de validation "
            "total score correspond au score sur la totalité du dataset"
            
            train_sc_a,train_sc_v=0,0
            val_sc_a,val_sc_v=0,0
            total_sc_a,total_sc_v=0,0
            
            random.shuffle(temp_shuffled_batches)
            
            # on commence par les batches d'entrainement 
            
            for k,batch_ in enumerate(temp_shuffled_batches[:N_train_batches]):
                
                "si les batches sont tout petits, on génère beaucoup de résultats"
                "on ne veut pas tous els sauvegarder: ça prend de la place et "
                " ça peut provoquer des fuites de RAM "
                
                "on se donne donc des indexes de sauvegarde; en l'occurrence"
                " tous les 20% du dataset "
                
                save_indexes=np.arange(0,N_train_batches,max(N_train_batches//5,1))
                
                write_this_step=(k in save_indexes) or ns=='all'
                
                if fit_strategy not in ('custom_gradient','scipy'):
                    print(" ERROR WRONG FIT STRATEGY !!!!")
                    break
                            
                
                "si on a choisi d'utiliser le gradient custom pour minimiser"
                if fit_strategy=="custom_gradient":
                
                    acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables)
                    
                    temp_id_variables={}
                    for i in id_variables:
                        temp_id_variables[i]=id_variables[i]


                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            temp_id_variables[key_]=id_variables[key_][batch_.index]

                    new_id_variables=propagate_gradient(jac_error_v if fit_on_v else jac_error_a,temp_id_variables)

                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            tparr=id_variables[key_]
                            tparr[batch_.index]=new_id_variables[key_]
                            new_id_variables[key_]=tparr

                    id_variables={}
                    for i in new_id_variables:
                        id_variables[i]=new_id_variables[i]
                    
                    
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
                        realvals[i]=prepare_dict(id_variables,k)[i]*scalers[i] if ('vw' not in i) else id_variables[i][0]*scalers[i]



                    if write_this_step:

                        saver(name="epoch_%i_batch_%i"%(n,k),save_path=spath,
                          id_variables=realvals,
                          train_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                          train_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                          val_sc_a=val_sc_a,
                          val_sc_v=val_sc_v,
                          total_sc_a=total_sc_a,
                          total_sc_v=total_sc_v)
                        
                        if wind_signal:
                            print([i for i in id_variables],[id_variables[key] for key in id_variables.keys()])
                            windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                            nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                            windsave_df.to_csv(os.path.join(spath,nsave))
                            del(windsave_df)
                        gc.collect()
                        
                        
                        
                        
                elif fit_strategy=="scipy":
                    
                    "si on a choisi d'utiliser scipy pour minimiser"
                    
                    temp_id_variables={}
                    for i in id_variables:
                        temp_id_variables[i]=id_variables[i]
                        
                    "il faut s'y prendre comme ceci pour copier un dict,"
                    "faire dict_B=dict_A fait que si on modifie B on modifie A"
                    

                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            temp_id_variables[key_]=id_variables[key_][batch_.index]
                            

                    X_start=dict_to_X(temp_id_variables)
                    #bnds=[bounds[i] for i in id_variables]


                    writtargs=[n,k,val_sc_a,val_sc_v,total_sc_a,total_sc_v,time.time(),write_this_step,temp_id_variables]
                       
                    sol_scipy=minimize(fun_cost_scipy,
                                       X_start,
                                       args=(batch_,scalers,writtargs),
                                        method="L-BFGS-B" if ns==-1 else 'SLSQP',
                                        jac=True)#,options={"maxiter":1})
                    
                    new_id_variables=X_to_dict(sol_scipy["x"],base_dict=temp_id_variables)
                    
                    if wind_signal:
                        for key_ in ('vw_i','vw_j'):
                            tparr=id_variables[key_]
                            print(key_,tparr,new_id_variables[key_])
                            tparr[batch_.index]=new_id_variables[key_]
                            new_id_variables[key_]=tparr

                    id_variables={}
                    for i in new_id_variables:
                        id_variables[i]=new_id_variables[i]
                    
                    if wind_signal and write_this_step:
                        windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                        nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                        windsave_df.to_csv(os.path.join(spath,nsave))
                        del(windsave_df)
                        
                    if write_this_step:
                        gc.collect()
                        
                        
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
            
            
            # on itère sur les batches de validation
            if ns!="all" and N_val_batches>0:
                for k,batch_ in enumerate(temp_shuffled_batches[N_train_batches:]):
        
                    save_indexes=np.arange(0,N_val_batches,max(1,N_val_batches//10))
                    write_this_step=(k in save_indexes) or ns=='all'

                    acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables)

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
                        realvals[i]=prepare_dict(id_variables,k)[i]*scalers[i] if ('vw' not in i) else id_variables[i][0]*scalers[i]
                        
                    if write_this_step:
                        saver(name="epoch_%i_batch_%i"%(n,k+N_train_batches),save_path=spath,
                          id_variables=realvals,
                          train_sc_a=train_sc_a,
                          train_sc_v=train_sc_v,
                          val_sc_a=np.sqrt(np.mean(square_error_a,axis=0)),
                          val_sc_v=np.sqrt(np.mean(square_error_v,axis=0)),
                          total_sc_a=total_sc_a,
                          total_sc_v=total_sc_v)
    
                        if wind_signal:
                            windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                            nsave="WINDSIG_epoch_%i_batch_%i"%(n,k)+".csv"
                            windsave_df.to_csv(os.path.join(spath,nsave))
                        gc.collect()
                        
            if N_val_batches!=0:
                val_sc_a/=N_val_batches
                val_sc_v/=N_val_batches
    
            acc_pred,speed_pred,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared,id_variables)
            total_sc_a=np.sqrt(np.mean(square_error_a,axis=0))
            total_sc_v=np.sqrt(np.mean(square_error_v ,axis=0))
            
            realvals={}
            for i in id_variables.keys():
                realvals[i]=prepare_dict(id_variables,0)[i]*scalers[i] if ('vw' not in i) else id_variables[i][0]*scalers[i]
    
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
            
            if wind_signal:
                windsave_df=pd.DataFrame(data=np.array([id_variables['vw_i'],id_variables['vw_j']]).T,columns=['w_i','w_j'])
                nsave="WINDSIG_epoch_%i"%(n)+".csv"
                windsave_df.to_csv(os.path.join(spath,nsave))
        return 0 
          
    train_loop(data_batches,id_variables,n_epochs=n_epochs)
    
    return 0

from multiprocessing import Pool

if __name__ == '__main__':
    
    blr_range=[0.5*10**i for i in range(-3,-2,1)]

    
    ns_range=[1.0]

    fit_arg_range=[False]

    x_r=[[i,j,k] for j in blr_range for i in  fit_arg_range  for k in ns_range ]

    # x_r.append([False, 'scipy', -1])

    # x_r.append([False, 'scipy', 1])
    # x_r.append([True, 'scipy',  1])
    
    # x_r.append([False, 'scipy', 5])
    # x_r.append([True, 'scipy',  5])
    
    # x_r.append([False, 'scipy', 'all'])
    # x_r.append([True, 'scipy',  'all'])

    
    print(x_r,len(x_r))

    pool = Pool(processes=len(x_r))
    alidhali=input('LAUNCH ? ... \n >>>>')
    pool.map(main_func, x_r)
# blr_range=[0.5*10**i for i in range(0,-5,-5)]


# ns_range=[1]

# fit_arg_range=[True]

# x_r=[[i,j,k] for j in blr_range for i in  fit_arg_range  for k in ns_range ]
# print(x_r,len(x_r))
    # main_func(x_r[0])
