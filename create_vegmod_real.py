import numpy as np
#import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.linalg import toeplitz
from scipy.spatial import distance
from joblib import Parallel, delayed
import pickle
import os
from static_1Dfunction import *




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




#Import parameter
tmax=20000
dt=0.1
L=100


n_mode=3
rain=0.8


path='noise_tmax_%.1f_dt_%.1f_L_%.1f'%(tmax,dt,L)


path_to_save='noise_tmax_%.1f_dt_%.1f_L_%.1f/realist/R_%.2f/n=%d'%(tmax,dt,L,rain,n_mode)
try:
    os.mkdir(path_to_save)
    print("Folder %s created!" % path)
except FileExistsError:
    print("Folder %s already exists" % path)





with open(path+"/Param.txt", "rb") as fp:   
    Param = pickle.load(fp)

with open(path+"/axes.txt", "rb") as fp:   
    axes = pickle.load(fp)

sigma=Param[0] #sigma of the biomass
sigma_O=[0.1]
lmb_t=Param[1] #lmb_t
lmb_s=Param[2] #lmb_s
t=axes[0]
x=axes[1]

n_sigma=np.shape(sigma)[0]
n_lmb_t=np.shape(lmb_t)[0]
n_lmb_s=np.shape(lmb_s)[0]

M=np.shape(t)[0]
N=np.shape(x)[0]

tmax=(t[1]-t[0])*M
L=(x[1]-x[0])*N

ind_lmb_t=0
ind_lmb_s=0
ind_sigma=0

n_real=15

#Create indices for the noise
ind_noise=np.zeros((3,n_real),dtype='int')
for i in range(n_real):
    ind_noise[:,i]=np.array([int(i),int((i+1)%15),int((i+2)%15)])


Sol=[]
#Spatial grid
L=100
N=100
dx=2*np.pi/N
x=np.arange(0,2*np.pi,dx)
#Temporal grid

M=int(tmax/dt)
Dt=0.1
t=np.arange(0,tmax,dt)



#precipitation
ind=selec_rain(rain,2,18,Rains_mode_tot[n_mode],np.mean(P_mode_tot[n_mode],axis=1))[0]
R=Rains_mode_tot[n_mode][ind]
prec=R+t*0 

P0=P_mode_tot[n_mode][ind]
W0=W_mode_tot[n_mode][ind]
O0=O_mode_tot[n_mode][ind]

print('tmax= %.1f'%(tmax))
print('Starting point')
print(n_mode)
print(R)


#lmb_t for the biomass
ind_lmb_t_P=2



for i in range(1,n_sigma):
    for j in range(n_lmb_t):
        for k in range(n_lmb_s):
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma_O[0],lmb_t[j]), "rb") as fp:
                Noise_O = np.array(pickle.load(fp)) 
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[i],lmb_t[ind_lmb_t_P],lmb_s[k]), "rb") as fp:
                Noise_P = np.array(pickle.load(fp))
            print('Runs for the following parameters')
            print('sigma=%.1f'%(sigma[i]))
            print('O noise: lmb_s=inf lmb_t=%.1f'%(lmb_t[j]))
            print('P noise: lmb_s=%.1f and lmb_t= %.1f'%(lmb_s[k],lmb_t[ind_lmb_t_P]))
            results =Parallel(n_jobs=n_real)(delayed(VegModelII_Riet_Spec_1D_02pi_noise_AR1)(L=L,N=N,M=M,tmax=tmax,dt=dt,Dt=Dt,prec=prec,P0=P0+np.random.randn(N)*0.01,W0=W0+np.random.randn(N)*0.01,O0=O0+np.random.randn(N)*0.01,param=param,P_noise=Noise_P[ind_noise[0,n],:,:],W_noise=Noise_P[ind_noise[1,n],:,:]*0,O_noise=Noise_O[ind_noise[2,n],:,:]) for n in range(n_real))
            print('---------------------------')
            print('Saving')
            with open(path_to_save+"/VegMod_P_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_O_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma[i],lmb_t[ind_lmb_t_P],lmb_s[k],sigma_O[0],lmb_t[j]), "wb") as fp:
                pickle.dump(results, fp)
            del results

'''
#Modif


ind_runs=[[1,0,4],[1,1,12],[1,2,4],[1,2,5],[1,2,8],[2,0,12],[2,2,10],[2,3,6],[2,5,0],[4,1,7],[4,2,5],[4,4,5],[5,0,7],[5,3,6],[5,6,6],[6,4,2],[7,4,8],[7,6,3],[8,4,4],[8,6,12],[9,1,12],[9,2,2],[9,2,3],[9,2,4],[9,2,6],[9,6,11]]

ind_runs=[[5,6,6],[6,4,2],[7,4,8],[7,6,3],[8,4,4],[8,6,12],[9,1,12],[9,2,2],[9,2,3],[9,2,4],[9,2,6],[9,6,11]]


ind_runs=[[0,0,6],[0,1,0],[0,5,11],[0,6,7]]

for i in range(len(ind_runs)):
    with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma_O[0],lmb_t[ind_runs[i][1]]), "rb") as fp:
        Noise_O = np.array(pickle.load(fp))
    with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[ind_runs[i][0]],lmb_t[ind_lmb_t_P],lmb_s[ind_runs[i][2]]), "rb") as fp:
        Noise_P = np.array(pickle.load(fp))
    print('Runs for the following parameters')
    print('sigma=%.1f'%(sigma[ind_runs[i][0]]))
    print('O noise: lmb_s=inf lmb_t=%.1f'%(lmb_t[ind_runs[i][1]]))
    print('P noise: lmb_s=%.1f and lmb_t= %.1f'%(lmb_s[ind_runs[i][2]],lmb_t[ind_lmb_t_P]))
    results =Parallel(n_jobs=n_real)(delayed(VegModelII_Riet_Spec_1D_02pi_noise_AR1)(L=L,N=N,M=M,tmax=tmax,dt=dt,Dt=Dt,prec=prec,P0=P0+np.random.randn(N)*0.01,W0=W0+np.random.randn(N)*0.01,O0=O0+np.random.randn(N)*0.01,param=param,P_noise=Noise_P[ind_noise[0,n],:,:],W_noise=Noise_P[ind_noise[1,n],:,:]*0,O_noise=Noise_O[ind_noise[2,n],:,:]) for n in range(n_real))
    print('---------------------------')
    print('Saving')
    with open(path_to_save+"/VegMod_P_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_O_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma[ind_runs[i][0]],lmb_t[ind_lmb_t_P],lmb_s[ind_runs[i][2]],sigma_O[0],lmb_t[ind_runs[i][1]]), "wb") as fp:
        pickle.dump(results, fp)
    del results


'''
