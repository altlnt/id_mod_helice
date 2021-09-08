

import  numpy as np
# %%   ####### PARAMETERS
import sys
import time
from sympy import Symbol, Matrix, symbols, sin, lambdify
import gc 
from collections import OrderedDict

"cette fonction est le main"
"on utilise le multiprocessing pour tester plusieurs metaparamètres"
"elle est appelée par le process"


def main_func(x):
    
    # récupération des arguments
    fit_arg,blr,ns=x[0],x[1],x[2]
    
    
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
    
    wind_signal=True
    assume_nul_wind=False # if true, le signal de vent constant vaut zéro
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
    
    log_path="./logs/vol12/log_real_processed.csv"
    save_dir_name="results"

    # Paramètres utilitaires
    
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
    
    vwi0=0.0
    vwj0=0.0
    
    
    cd0sa_0 = 0.010
    cd0fp_0 = 0.010
    cd1sa_0 = 4.55 
    cl1sa_0 = 5 
    cd1fp_0 = 2.5 
    coeff_drag_shift_0= 0.5 
    coeff_lift_shift_0= 0.05 
    coeff_lift_gain_0= 2.5
    
    
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
    vwj0]
    
    
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
            "[mass,Area,r,rho,kv_motor,pwmmin,pwmmax,U_batt,b1,c10,c20,c30,ch10,ch20,di0,dj0,dk0,vwi0,vwj0,kt0]":physical_params,
            "bounds":bounds}
    
    
    
    print(" META PARAMS \n")
    [print(i,":", metap[i]) for i in metap.keys()]
    
    
    
    # %%   ####### Saving function
    
    import os
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
    
    import pandas as pd
    
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
        
    t0=time.time()
    
    def Generate_equation(used_logged_v_in_model=used_logged_v_in_model):
          vlog_i,vlog_j,vlog_k=symbols("vlog_i,vlog_j,vlog_k",real=True)
          vpred_i,vpred_j,vpred_k=symbols("vpred_i,vpred_j,vpred_k",real=True)    
          v_i,v_j,v_k=(vlog_i,vlog_j,vlog_k) if used_logged_v_in_model else (vpred_i,vpred_j,vpred_k)
          v=Matrix([[v_i],
             [v_j],
             [v_k]])
          
          print("\nElapsed : %f s , Prev step time: -1 s \\ Generating first symbols ..."%(time.time()-t0))
          dt=Symbol('dt',positive=True,real=True)
          CL_1_sa = Symbol('C^{sa}_{L,1}',real=True)             # Coeff rechercher
          CD_0_sa = Symbol('C^{sa}_{D,0}',real=True)             # Coeff rechercher
          CD_1_sa = Symbol('C^{sa}_{D,1}',real=True)             # Coeff rechercher
          CD_0_fp = Symbol('C^{fp}_{D,0}',real=True)             # Coeff rechercher
          CD_1_fp = Symbol('C^{fp}_{D,1}',real=True)             # Coeff rechercher                      
          k_0 = Symbol('k_0', real=True)                         # coeff rechercher
          k_1 = Symbol('k_1', real=True)                         # coeff rechercher
          k_2 = Symbol('k_2', real=True)                         # coeff rechercher
          delta_s = Symbol('delta_s', real=True)                # Coeff rechercher : largeur du stall
          alpha_s = Symbol('alpha_s',real=True)                 # Coeff rechercher
          B_B       = Matrix([[1,0,0], [0,1,0], [0,0,1]])                              # Base dans le repère body
          omega1, omega2, omega3 = symbols('\omega_1, \omega_2, \omega_3', real=True)
          Omega     = Matrix([omega1, omega2, omega3])                                 # Vecteur de rotation
          r00, r01, r02, r10, r11, r12, r20, r21, r22 = symbols('r_{00}, r_{01}, r_{02}, r_{10}, r_{11}, r_{12}, r_{20}, r_{21}, r_{22}', real=True)
          R         = Matrix([[r00,r01,r02], [r10,r11, r12], [r20, r21, r22]])          # Matrice de rotation
          Vb1,Vb2,Vb3=symbols('V_{b1} V_{b2} V_{b3}',real=True)
          v_B        = Matrix([Vb1, Vb2, Vb3])                                          # Vitesse du corps (repère drone)
          Vw1,Vw2,Vw3=symbols('V_{w1} V_{w2} V_{w3}',real=True)                         # Vitesse du vent dans le repère NED 
          v_W        = Matrix([Vw1, Vw2, Vw3])
          xcp, ycp, zcp = symbols('x_{cp}, y_{cp}, z_{cp}')
          X_cp     = Matrix([xcp, ycp, zcp])                                            # Position du centre de poussé d'un corps dans le repère body
          C_t, C_q, C_h=symbols('C_t,C_q, C_h',real=True)                           # Coefficient de poussée des moteurs, coefficient de couple des moteurs
          motor_axis_in_body_frame = Matrix([1,0,0])                                    # Axe des moteurs, ici placé en mode avion
          omega_rotor = symbols('\omega_{rotor}', real=True)                            # Vitesse de rotation des moteurs
          crossward_B = B_B[:,1]
          c45, s45 = np.cos(45*np.pi/180), np.sin(45*np.pi/180)
          r = np.array(((1,0, 0),(0,c45,-s45),(0,s45, c45)))
          r_neg     = np.array(((1,0, 0), (0,c45, s45),(0,-s45, c45)))
          R_list_sympy = [R, R,  R * r,  R *r_neg, R]      # Liste des matrices de rotation de chaque surface portante du drone, seul les éléments de la queue (element 3 et 4) ne sont pas dans le repère inertiel. 
          cp1x,cp1y, cp1z, cp2x,cp2y,cp2z, cp3x,cp3y,cp3z, cp4x,cp4y,cp4z,cp5x,cp5y,cp5z = symbols('cp1x,cp1y, cp1z, cp2x,cp2y,cp2z, cp3x,cp3y,cp3z, cp4x,cp4y,cp4z,cp5x,cp5y,cp5z', real=True)
          cp_list = [Matrix([cp1x,cp1y, cp1z]), Matrix([cp2x,cp2y,cp2z]), Matrix([cp3x,cp3y,cp3z]), Matrix([cp4x,cp4y,cp4z]), Matrix([cp5x,cp5y,cp5z])]
          A1, A2, A3 = symbols('A_1 A_2 A_3', real=True)
          Aire_list = [A1, A1, A2, A2, A3]  # Liste des 1/2 * rho * S pour chaque surface
          cp_list_rotor = [Matrix([0.713,0.475,0]), Matrix([0.713,-0.475,0])]
          spinning_sense_list = [1,-1]
          
          ##### Listes des angles (d'attaque et de contrôles) pour faire la somme des forces en une seule équations
          alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha_1, alpha_2, alpha_3, alpha_4, alpha_5', real=True)
          alpha_list =Matrix([alpha1, alpha2, alpha3, alpha4, alpha5])
          alpha0_1, alpha0_2, alpha0_3, alpha0_4, alpha0_5 = symbols('alpha_0_1, alpha_0_2, alpha_0_3, alpha_0_4, alpha_0_5', real=True)
          alpha0_list = Matrix([alpha0_1, alpha0_2, alpha0_3, alpha0_4, alpha0_5])
          delta0_1, delta0_2, delta0_3, delta0_4, delta0_5 = symbols('delta_0_1, delta_0_2, delta_0_3, delta_0_4, delta_0_5', real=True)
          delta0_list = Matrix([delta0_1, delta0_2, delta0_3, delta0_4, delta0_5])
              
          t1=time.time()
          print("Elapsed : %f s , Prev step time: -1 s \\ Generating dynamics function ..."%(t1-t0))
          def compute_alpha(dragDirection, liftDirection, frontward_Body, VelinLDPlane):
              calpha= np.vdot(dragDirection, frontward_Body)
              absalpha= -np.arccos(calpha)
              signalpha = np.sign(np.vdot(liftDirection, frontward_Body)) 
              if np.linalg.norm(VelinLDPlane)>1e-7 :
                  alpha = signalpha*absalpha 
              else :
                  alpha=0
              if abs(alpha)>0.5*np.pi:
                  if alpha>0 :alpha=alpha-np.pi 
                  else: alpha=alpha+np.pi         
              return alpha
      
        ##################################################### génération des équations pour Cd et Cl (utiliser pour générer les équations symbolique pour chaque surface portantes) ####################################################
          def compute_cl_cd(a, a_0, a_s, d_0, d_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, cd0sa, cd1sa):
          
              CL_sa = 1/2 * cl1sa * sin(2*(a + (k1*d_0) + a_0))
              CD_sa = cd0sa + cd1sa * sin(a + (k0*d_0) + a_0)*sin(a +  (k0*d_0) + a_0)
          
              CL_fp = 1/2 * cd1fp * sin(2*(a+ (k1*d_0) + a_0))
              CD_fp = cd0fp + cd1fp * sin(a + (k0*d_0) + a_0)*sin(a +  (k0*d_0) + a_0)
          
              puiss=5
              s = 1.0 - ((a+a_0)**2/a_s**2)**puiss/(((a+a_0)**2/a_s**2)**puiss + 100+200*d_s)
          
              C_L = CL_fp + s*(CL_sa - CL_fp) + k2 * sin(d_0)
              C_D = CD_fp + s*(CD_sa - CD_fp)
              return C_L, C_D
           
          def GenDirectForceWing(Omega, cp, vB, vW, R, crossward_Body):
               # Cette fonction permet d'obtenir les directions des efforts de portances et de trainé en fonction des vitesses, et de l'orientation dans le repère NED.
               Air_speed_earth = vB - vW 
               Air_speed_body  = (R.T* Air_speed_earth) - cp.cross(Omega)
               VelinLDPlane    = Air_speed_body - Air_speed_body.dot(crossward_Body.T) * crossward_Body
               
               dragDirection = -VelinLDPlane / VelinLDPlane.norm()  #if VelinLDPlane_norm > VelLim else Matrix([0,0,0])
               liftDirection = -crossward_Body.cross(dragDirection) #if crossward_NED.norm() > VelLim else Matrix([0,0,0])
           
               return VelinLDPlane, dragDirection, liftDirection
           
          def GenForceWing(A, VelinLDPlane, dragDirection, liftDirection, Cd, Cl, cp):
              # Cette fonction permet de générer les forces aerodynamique d'une aile dans son repère.
              D = A * VelinLDPlane.norm()**2 * dragDirection * Cd
              L = A * VelinLDPlane.norm()**2 * liftDirection * Cl
          
              F_wing = L+D 
              Torque_wing =  cp.cross(F_wing)
          
              return F_wing, Torque_wing
          
          def Generate_Sum_Force_wing(A_list, Omega, cp_list, R_list, vB, vW,  Cd_list, Cl_list, crossward_body, r_queue, r_queue_neg):
              # Cette function permet de généer l'équation complète de la somme des forces pour les différentes surfaces portantes 
              p = 0
              Sum_Force_Wing = Matrix([0,0,0])
              Sum_Torque_Wing =  Matrix([0,0,0])
              for i in cp_list:
                  VelinLDPlane, dragDirection, liftDirection= GenDirectForceWing(Omega, i, vB, vW, R_list[p], crossward_body)
                  if p == 2 :
                      # Comme la fonction GenForceWing donne les efforts des ailes dans leur repère propre, on doit passer par les matrice de rotation pour les ailes de la queue
                      F_wing, Torque_wing =  GenForceWing(A_list[p], VelinLDPlane, dragDirection, liftDirection, Cd_list[p], Cl_list[p], Matrix([0,0,0]))
                      Sum_Force_Wing  = Sum_Force_Wing +  r_queue.T  @ F_wing 
                      Sum_Torque_Wing = Sum_Torque_Wing + i.cross(r_queue.T  @ F_wing) 
                  elif p == 3 :
                      F_wing, Torque_wing =  GenForceWing(A_list[p], VelinLDPlane, dragDirection, liftDirection, Cd_list[p], Cl_list[p], Matrix([0,0,0]))
                      Sum_Force_Wing  = Sum_Force_Wing +  r_queue_neg.T  @ F_wing 
                      Sum_Torque_Wing = Sum_Torque_Wing + i.cross(r_queue_neg.T  @ F_wing) 
                  else:
                      F_wing, Torque_wing =  GenForceWing(A_list[p], VelinLDPlane, dragDirection, liftDirection, Cd_list[p], Cl_list[p], i)
                      Sum_Force_Wing  = Sum_Force_Wing +  F_wing
                      Sum_Torque_Wing = Sum_Torque_Wing + Torque_wing
                  p+=1
          
              return Sum_Force_Wing, Sum_Torque_Wing
          
          
          def GenForceMoteur(Omega, ct, cq, omega_rotor, cp, vB, vW, ch, R, motor_axis_in_body_frame, spinning_sense):
              ## Cette fonction calcule les effort produit par un rotor sur le drone en fonction de son sens de rotation et de sa localisation, les efforts sont donnés
              ## dans le repère inertiel. l'axe des moteur est placé suivant l'axe x du drone (mode avion seulement)
              Air_speed_earth = vB - vW
              air_speed_in_rotor_frame = (R.T* Air_speed_earth) - cp.cross(Omega)
              Axial_speed = air_speed_in_rotor_frame.dot(motor_axis_in_body_frame)
              lat_speed = air_speed_in_rotor_frame - (Axial_speed * (motor_axis_in_body_frame))
                  
              T = ct*omega_rotor**2
              H = ch * omega_rotor
              
              T_vec = T * motor_axis_in_body_frame - H * lat_speed
              
              torque = - omega_rotor * cq * lat_speed
              torque = - spinning_sense * cq * T * motor_axis_in_body_frame 
              torque_at_body_center = torque + cp.cross(T_vec.T)
                  
              return T_vec, torque_at_body_center
          
          def Generate_Sum_Force_Moteur(Omega, ct, cq, omega_rotor, cp_list_rotor, vB, vW, ch, R, motor_axis_in_body_frame_list, spinning_sense_list):
              # Calcul des forces des moteurs sur le drone, génère toutes les forces, ainsi que le couple appliqué au centre de gravité du drone, dans le repère inertiel
              p = 0
              Sum_Force_Rotor = Matrix([0,0,0])
              Sum_Torque_Rotor =  Matrix([0,0,0])
              for cp in cp_list_rotor:
                  F_rotor, Q_rotor = GenForceMoteur(Omega, ct, cq, omega_rotor, cp, vB, vW, ch, R, motor_axis_in_body_frame_list, spinning_sense_list[p])        
                  Sum_Force_Rotor  = Sum_Force_Rotor + F_rotor
                  Sum_Torque_Rotor = Sum_Torque_Rotor + Q_rotor
                  p+=1
          
              return Sum_Force_Rotor, Sum_Torque_Rotor
      
          def Compute_list_coeff(alpha_list, alpha_0_list, alpha_s, delta_0_list, delta_s, CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa):
              Cd_list = Matrix([0 for i in range(len(alpha_list))])
              Cl_list = Matrix([0 for i in range(len(alpha_list))])
              for count, alpha in enumerate(alpha_list):
                  Cl_list[count],  Cd_list[count] =  compute_cl_cd(alpha, alpha_0_list[count], alpha_s, delta_0_list[count], delta_s, CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa)
              return Cl_list , Cd_list
          
          "liftdrag forces"
          t2=time.time()
          print("Elapsed : %f s , Prev step time: %f s \\ Solving lifrdrag model ..."%(t2-t0,t2-t1))
          # Ces équation servent uniquement pour le calcul de l'angle d'attaque pour le calcul des coeffficient aérodynamique Cd et Cl, il n'interviennent pas directement dans le calcul des efforts. 
          VelinLDPlane, dragDirection, liftDirection=  GenDirectForceWing(Omega, X_cp, v_B, v_W, R, crossward_B)                        
          Cl_list, Cd_list = Compute_list_coeff(alpha_list, alpha0_list, alpha_s, delta0_list, delta_s, CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa)
          Sum_F_wing_complete, Sum_T_wing_complete = Generate_Sum_Force_wing(Aire_list, Omega, cp_list, R_list_sympy, v_B, v_W, Cd_list, Cl_list, crossward_B, r, r_neg)
          
          t3=time.time()
          print("Elapsed : %f s , Prev step time: %f s \\ Solving rotor model ..."%(t3-t0,t3-t2))
          ##################### Sommes des efforts des moteurs
          Sum_F_rotor_complete, Sum_T_rotor_complete = Generate_Sum_Force_Moteur(Omega, C_t, C_q, omega_rotor, cp_list_rotor, v_B, v_W, C_h, R, motor_axis_in_body_frame, spinning_sense_list)
          Sum_F_rotor_complete.simplify()
          Sum_T_rotor_complete.simplify()
          Effort_Aero_complete = [Sum_F_wing_complete + Sum_F_rotor_complete , Sum_T_wing_complete + Sum_T_rotor_complete]
          
          t35=time.time()
          print("Elapsed : %f s , Prev step time: %f s \\ Solving Dynamics ..."%(t35-t0,t35-t3))

          theta = OrderedDict(sorted(id_variables_sym.items()))
          
          
          Grad_Force_Aero_complete = Matrix([(Effort_Aero_complete[0]).jacobian([i for i in theta.values()])])
          Grad_Torque_Aero_complete = Matrix([(Effort_Aero_complete[1]).jacobian([i for i in theta.values()])])
          Grad_Effort_Aero_complete = [Grad_Force_Aero_complete,Grad_Torque_Aero_complete]
      
          ########## Equation du gradient utilisé en simulation ####################
          VelinLDPlane_function = lambdify((Omega, X_cp, v_B, v_W, R), VelinLDPlane, 'numpy')
          dragDirection_function = lambdify((Omega, X_cp, v_B, v_W, R), dragDirection, 'numpy')
          liftDirection_function = lambdify((Omega, X_cp, v_B, v_W, R), liftDirection, 'numpy')
          Effort_Aero_complete_function = lambdify((Aire_list, Omega, R, v_B, v_W, cp_list, alpha_list, alpha0_list, alpha_s, delta0_list, delta_s, CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa, C_t, C_q, C_h, omega_rotor), Effort_Aero_complete, 'numpy')
          Grad_Effort_Aero_complete_function = lambdify((Aire_list, Omega, R, v_B, v_W, cp_list, alpha_list, alpha0_list, alpha_s, delta0_list, delta_s, \
                                                         CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa, C_t, C_q, C_h, omega_rotor), Grad_Effort_Aero_complete, 'numpy')
          
          t37=time.time()
          print("Elapsed : %f s , Prev step time: %f s \\ Generating costs ..."%(t37-t0,t37-t35))
          
          m=symbols('m', real=True)
          g1,g2,g3=symbols('g1,g2,g3', real=True)
          g=Matrix([g1,g2,g3])
          
          w1,w2,w3,w4,w5,w0 = symbols('w_1,w_2,w_3,w_4,w_5,w_0')
      
          #Génération des équations finales pour la gradient du cout et des RMS error
          forces = R@(Sum_F_wing_complete + Sum_F_rotor_complete) + m*g
          # torque = R@(Sum_T_wing_complete + Sum_T_rotor_complete)
          
          new_acc = (forces/m)
          new_v = v + new_acc*dt
          
          vlog_i,vlog_j,vlog_k=symbols("vlog_i,vlog_j,vlog_k",real=True)
          v_log=Matrix([[vlog_i],[vlog_j],[vlog_k]])         
          
          alog_i,alog_j,alog_k=symbols("alog_i,alog_j,alog_k",real=True)
          alog=Matrix([[alog_i],[alog_j],[alog_k]])
         
          err_a=Matrix(alog-new_acc)
          err_v=Matrix(v_log-new_v)
          
          cost_scaler_a=symbols('C_sa',real=True,positive=True)
          cost_scaler_v=symbols('C_sv',real=True,positive=True)
          
          sqerr_a=Matrix([1.0/cost_scaler_a*(err_a[0,0]**2+err_a[1,0]**2+err_a[2,0]**2)])
          sqerr_v=Matrix([1.0/cost_scaler_v*(err_v[0,0]**2+err_v[1,0]**2+err_v[2,0]**2)])
          
          Ja=sqerr_a.jacobian([i for i in theta.values()])
          Jv=sqerr_v.jacobian([i for i in theta.values()])
          
          Y=Matrix([new_acc,new_v,sqerr_a,sqerr_v,Ja.T,Jv.T])
          
          X =(alog,v_log,dt, Aire_list, Omega, R, v_B, v_W, cp_list, alpha_list, alpha0_list, alpha_s, delta0_list, delta_s, \
              CL_1_sa, CD_1_fp, k_0, k_1, k_2, CD_0_fp, CD_0_sa, CD_1_sa, C_t, C_q, C_h, omega_rotor, \
                  g, m, w0,w1,w2,w3,w4,w5)
          
          
          model_func=lambdify(X,Y, modules='numpy')

          return model_func
          # CI DESSOUS : on spécifie quelles variables sont les variables d'identif
   
    model_func = Generate_equation(used_logged_v_in_model)
    # CI DESSOUS : on spécifie quelles variables sont les variables d'identif
    
    t7=time.time()
    # print("Elapsed : %f s , Prev step time: %f s \\ Done ..."%(t7-t0,t7-t6))
    
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
    import pandas as pd
    
    raw_data=pd.read_csv(log_path)
    
    print("PROCESSING DATA...")
    
    
    prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) or ("joy" in i)) ])
    prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
    
    
    # on choisit tmin et tmax selon quel log on utilise
    
    if "vol12" in log_path:
        tmin,tmax=(-1,1e10) 
    elif "vol1" in log_path:
        tmin,tmax=(41,265) 
    elif "vol2" in log_path:
        tmin,tmax=(10,140) 

        
    prep_data=prep_data[prep_data['t']>tmin]
    prep_data=prep_data[prep_data['t']<tmax]
    prep_data=prep_data.reset_index()
    
    
    for i in range(3):
        prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
        
        
    prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
    prep_data['t']-=prep_data['t'][0]
    prep_data=prep_data.drop(index=[0,len(prep_data)-1])
    prep_data=prep_data.reset_index()
    
    data_prepared=prep_data[:len(prep_data)]
    
    
    
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

        data_batches=[i.drop(columns=[j for j in data_prepared.keys() if (("level" in j ) or ("index") in j) ]) for i in np.array_split(data_prepared, N_minibatches)]

        data_batches=[i.reset_index() for i in data_batches]
        
        N_train_batches=round(train_proportion*N_minibatches)
        N_val_batches=N_minibatches-N_train_batches
    print("DATA PROCESS DONE")
        
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
                      "vw_j":vwj0}
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
    
    def arg_wrapping(batch,id_variables,scalers,data_index,speed_pred_previous):
        
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
        
        cost_scaler_v=1.0
        cost_scaler_a=1.0
    
        dt=min(batch['dt'][i],1e-2)

        vlog_i,vlog_j,vlog_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        vpred_i,vpred_j,vpred_k=speed_pred_previous 
        alog_i,alog_j,alog_k=batch['acc_ned_grad[0]'][i],batch['acc_ned_grad[1]'][i],batch['acc_ned_grad[2]'][i]
        vnext_i,vnext_j,vnext_k=batch['speed[0]'][i],batch['speed[1]'][i],batch['speed[2]'][i]
        
        m=non_id_variables['m'] if 'm' in non_id_variables else id_variables['m']
        
        vw_i=non_id_variables['vw_i'] if 'vw_i' in non_id_variables else id_variables['vw_i']
        vw_j=non_id_variables['vw_j'] if 'vw_j' in non_id_variables else id_variables['vw_j']
        
        if wind_signal:
            vw_i,vw_j=vw_i[i],vw_j[i]
        
        
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
        
        cd0sa=non_id_variables['cd0sa'] if 'cd0sa' in non_id_variables else id_variables['cd0sa']
        cd0fp=non_id_variables['cd0fp'] if 'cd0fp' in non_id_variables else id_variables['cd0fp']
        cd1sa=non_id_variables['cd1sa'] if 'cd1sa' in non_id_variables else id_variables['cd1sa'],
        cl1sa=non_id_variables['cl1sa'] if 'cl1sa' in non_id_variables else id_variables['cl1sa']
        cd1fp=non_id_variables['cd1fp'] if 'cd1fp' in non_id_variables else id_variables['cd1fp']
        coeff_drag_shift=non_id_variables['coeff_drag_shift'] if 'coeff_drag_shift' in non_id_variables else id_variables['coeff_drag_shift']
        coeff_lift_shift=non_id_variables['coeff_lift_shift'] if 'coeff_lift_shift' in non_id_variables else id_variables['coeff_lift_shift']
        coeff_lift_gain=non_id_variables['coeff_lift_gain'] if 'coeff_lift_gain' in non_id_variables else id_variables['coeff_lift_gain']
        
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
    
        "si n est la taille du batch"
        "cette fonction sert à appeler n fois la fonction lambdifiée"
        " de sympy "
        
        "on obtient n acc prédites, n vitesses prédites, n jacobiennes...."
        
    
        "batch est un dataframe"
        "id_variables sont les variables d'identification"
        "scalers sont les coefficients de mise à l'échelle"
        
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
            
    

        return acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v
    

    
    # %%   Gradient
    import random
    

    
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

        acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch,id_var,scalers)
        
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
                
                    acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(batch_,id_variables,scalers)
                    
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
    
            acc_pred,speed_pred,omegas,square_error_a,square_error_v,jac_error_a,jac_error_v=pred_on_batch(data_prepared,id_variables,scalers)
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
    
    blr_range=[0.5*10**i for i in range(0,-5,-5)]

    
    ns_range=[-1]

    fit_arg_range=[True]

    x_r=[[i,j,k] for j in blr_range for i in  fit_arg_range  for k in ns_range ]

    # x_r.append([False, 'scipy', -1])

    # x_r.append([False, 'scipy', 1])
    # x_r.append([True, 'scipy',  1])
    
    # x_r.append([False, 'scipy', 5])
    # x_r.append([True, 'scipy',  5])
    
    # x_r.append([False, 'scipy', 'all'])
    # x_r.append([True, 'scipy',  'all'])

    
    print(x_r,len(x_r))

    # pool = Pool(processes=len(x_r))
    # alidhali=input('LAUNCH ? ... \n >>>>')
    # pool.map(main_func, x_r)
blr_range=[0.5*10**i for i in range(0,-5,-5)]


ns_range=[-1]

fit_arg_range=[True]

x_r=[[i,j,k] for j in blr_range for i in  fit_arg_range  for k in ns_range ]
print(x_r,len(x_r))
main_func(x_r[0])
