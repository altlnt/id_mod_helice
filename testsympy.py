#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:26:59 2021

@author: alex
"""

import time
from sympy import *

    
"regresion on a"
fit_on_v=True
used_logged_v_in_model=not fit_on_v

model_motor_dynamics=True

n_epochs=30



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
    
    