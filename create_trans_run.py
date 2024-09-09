import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from static_1Dfunction import *
from joblib import Parallel, delayed



def trans_rainfall(t,t_i,t_f,R_i,R_f):
    R=np.zeros(np.shape(t))
    for i in range(np.shape(t)[0]):
        if t[i]<=t_i:
            R[i]=R_i
        elif t[i]>t_i and t[i]<t_f:
            R[i]=(R_f-R_i)/(t_f-t_i)*(t[i]-t_i)+R_i
        else:
            R[i]=R_f
    return(R,(R_f-R_i)/(t_f-t_i))


#import
dir_name=['hom','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=1alt','n=2bis','n=3bis_1','n=3bis_2','n=4bis_1','n=4bis_2','n=4bis_3']
name_mode=['Homogeneous solution','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=1 alternative','n=2 bis','n=3 bis_1','n=3 bis_2','n=4 bis_1','n=4bis_2','n=4bis_3']

P_mode_tot=[]
W_mode_tot=[]
O_mode_tot=[]
Rains_mode_tot=[]
Stab_mode_tot=[]
Lmb_mode_tot=[]

for i in range(len(dir_name)):
    #print('Saving files for '+dir_name[i])
    with open("L100/"+dir_name[i]+"/P_mode_tot.txt", "rb") as fp:
        P_mode_tot.append(pickle.load(fp))
    with open("L100/"+dir_name[i]+"/W_mode_tot.txt", "rb") as fp:
        W_mode_tot.append(pickle.load(fp))
    with open("L100/"+dir_name[i]+"/O_mode_tot.txt", "rb") as fp:
        O_mode_tot.append(pickle.load(fp))
    with open("L100/"+dir_name[i]+"/Rains_mode_tot.txt", "rb") as fp:
        Rains_mode_tot.append(pickle.load(fp))
    with open("L100/"+dir_name[i]+"/Stab_mode_tot.txt", "rb") as fp:
        Stab_mode_tot.append(pickle.load(fp))
    with open("L100/"+dir_name[i]+"/Lmb_mode_tot.txt", "rb") as fp:
        Lmb_mode_tot.append(pickle.load(fp))
         
   
with open("L100/param.txt", "rb") as fp:   
    param = pickle.load(fp)


def create_noisy_trans_run(rate):
    eps=0.0001
    n=0
    ind=selec_rain(rain_i,1,9,Rains_mode_tot[n_mode],np.mean(P_mode_tot[n_mode],axis=1))[0]
    #Spatial grid
    L=100
    N=100
    dx=2*np.pi/N
    x=np.arange(0,2*np.pi,dx)
    #Temporal grid
    tmax=300000
    M=tmax*100+1
    dt=tmax/(M-1)
    Dt=10
    t=np.linspace(0,tmax,M)
    #precipitation
    t_i=10000
    t_f=t_i+(rain_f-rain_i)/rate
    prec,rate=trans_rainfall(t,t_i,t_f,rain_i,rain_f)
    #initial condition
    lmb,vec=stability_eigen(P_mode_tot[n_mode][ind],W_mode_tot[n_mode][ind],O_mode_tot[n_mode][ind],Rains_mode_tot[n_mode][ind],L,param)
    P0=P_mode_tot[n_mode][ind]*1+eps*np.real(vec[:N,n])
    W0=W_mode_tot[n_mode][ind]*1+eps*np.real(vec[N:2*N,n])
    O0=O_mode_tot[n_mode][ind]*1+eps*np.real(vec[2*N:,n])
    P_full,W_full,O_full,R_full,t=VegModelII_Riet_Spec_1D_02pi(L,N,M,tmax,dt,Dt,prec,P0,W0,O0,param)
    Sol_Het=[P_full,W_full,O_full,R_full,t,rate]
    return(Sol_Het)


rain_i=1.2
rain_f=0.65
n_mode=2
#Rate=-np.array([0.000005,0.00001,0.00005])
Rate=-np.array([0.000004,0.000005,0.000006,0.000008,0.00002,0.00003,0.0001,0.001])

n_rate=np.shape(Rate)[0]

import multiprocessing as mp


results =Parallel(n_jobs=n_rate)(delayed(create_noisy_trans_run)(Rate[n]) for n in range(n_rate))    




path = 'trans_R_%.2f_%.2f_n=%d'%(results[0][3][0],results[0][3][-1],n_mode)

try:
    os.mkdir(path)
    print("Folder %s created!" % path)
except FileExistsError:
    print("Folder %s already exists" % path)



#with open("noise_R_%.2f_n=%d/homogeneous.txt"%(R_full[0],n_mode), "wb") as fp:
#    pickle.dump(Hom, fp)

with open(path+"/trans.txt", "wb") as fp:
    pickle.dump(results, fp)


