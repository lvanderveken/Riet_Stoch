import pickle
import numpy as np
#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from mpl_toolkits.mplot3d import Axes3D
from static_1Dfunction import *
from IPython.display import display, HTML
from matplotlib import cm
import seaborn
import matplotlib
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed



path='sensi_param/'


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

dx=2*np.pi/100
L=100
x=np.arange(0,2*np.pi,dx)*(L/(2*np.pi))


N_mode=len(P_mode_tot)
print(N_mode)

def detect_bumps(x,treshold):
    NZeros=np.zeros(np.shape(x)[0])
    for i in range(int(np.shape(x)[0])):
        if x[i]>(np.min(x)+treshold):
            NZeros[i]=1
    bumps=[]
    on=0
    for i in range(int(np.shape(x)[0])):
        if NZeros[i]==1 and on==0:
            bumps.append([])
            bumps[-1].append(i)
            on=1
        elif NZeros[i]==1 and on==1:
            bumps[-1].append(i)
        else:
            on=0
    if NZeros[0]==1 and NZeros[-1]==1:
        bumps[0]+=bumps[-1]
        bumps=bumps[:-1]
    return(bumps)


# Colorblind
color_mode=[]
color_mode.append('black')
seaborn.color_palette("colorblind")
for i in range(10):
    color_mode.append(seaborn.color_palette("colorblind")[i])

color_mode.append(seaborn.color_palette("dark")[1])
color_mode.append(seaborn.color_palette("dark")[2])
color_mode.append(seaborn.color_palette("dark")[3])
color_mode.append(seaborn.color_palette("dark")[4])
color_mode.append(seaborn.color_palette("dark")[5])
color_mode.append(seaborn.color_palette("dark")[6])

param2=copy.copy(param)
f_Dw=0.9
f_Dp=0.9
f_Do=1
param2['DP']=f_Dp*param['DP']
param2['DW']=f_Dw*param['DW']
param2['DO']=f_Do*param['DO']

#param2['c']=100*param['c']
#param2['gmax']=100*param['gmax']
#param2['d']=10*param['d']
#print(param['DP'])
#print(param2['DP'])


n_mode=4
rains=np.linspace(0.8,1.1,5)
ind_rains=np.zeros(np.shape(rains)[0],dtype=int)
Sol=[]
#Spatial grid
L=100
N=100
dx=2*np.pi/N
x=np.arange(0,2*np.pi,dx)
#Temporal grid
tmax=5000
M=tmax*10+1
dt=tmax/(M-1)
Dt=0.1
t=np.linspace(0,tmax,M)


ind=selec_rain(0.75,2,18,Rains_mode_tot[n_mode],np.mean(P_mode_tot[n_mode],axis=1))[0]
for i in range(np.shape(rains)[0]):
    t=np.linspace(0,tmax,M)
    rain=rains[i]
    eps=0.01
    n=0
    ind=selec_rain(rain,np.mean(P_mode_tot[n_mode][ind]),18,Rains_mode_tot[n_mode],np.mean(P_mode_tot[n_mode],axis=1))[0]
    ind_rains[i]=ind
print(ind_rains)
#ind=selec_rain(0.5,2,18,Rains_mode_tot[n_mode],np.mean(P_mode_tot[n_mode],axis=1))[0]
for i in range(np.shape(ind_rains)[0]):
    t=np.linspace(0,tmax,M)
    #precipitation
    R=Rains_mode_tot[n_mode][ind_rains[i]]
    prec=R+t*0   
    #initial condition     
    Bumps=detect_bumps(P_mode_tot[n_mode][ind_rains[i]],0.01)
    P0=copy.copy(P_mode_tot[n_mode][ind_rains[i]])
    #P0[Bumps[-1]]=0
    W0=W_mode_tot[n_mode][ind_rains[i]]
    O0=O_mode_tot[n_mode][ind_rains[i]]
    P_full,W_full,O_full,R_full,t=VegModelII_Riet_Spec_1D_02pi(L,N,M,tmax,dt,Dt,prec,P0+0*np.random.rand(N),W0+0*np.random.rand(N),O0+0*np.random.rand(N),param2)
    Sol.append([P_full,W_full,O_full,R_full,t])

  

n_rains=np.shape(rains)[0]
t=Sol[0][-1]


index_t_stab=np.zeros(n_rains,dtype=int)
for i in range(n_rains):
    j=100
    diff=100
    while diff>10**(-4):
        diff=np.max(np.abs(Sol[i][0][j+1,:]-Sol[i][0][j,:]))
        j=j+1
    index_t_stab[i]=int(j)


print(index_t_stab)



plt.rc('font', size=14)
n_rains=np.shape(rains)[0]
t=Sol[0][-1]


Init_sol_P=np.zeros((n_rains,N))
Init_sol_W=np.zeros((n_rains,N))
Init_sol_O=np.zeros((n_rains,N))

for i in range(n_rains):
    Init_sol_P[i,:]=Sol[i][0][-1,:]
    Init_sol_W[i,:]=Sol[i][1][-1,:]
    Init_sol_O[i,:]=Sol[i][2][-1,:]



print('Now we remove one patch')


def run_with_removed_patch(Init_sol_P,Init_sol_W,Init_sol_O,r):
    #Spatial grid
    L=100
    N=100
    dx=2*np.pi/N
    x=np.arange(0,2*np.pi,dx)
    #Temporal grid
    tmax=10**(6)
    M=tmax*1+1
    dt=tmax/(M-1)
    Dt=10
    t=np.linspace(0,tmax,M)
    #precipitation
    #precipitation
    R=Rains_mode_tot[n_mode][ind_rains[r]]
    prec=R+t*0
    #initial condition
    Bumps=detect_bumps(Init_sol_P[i,:],0.1)
    P0=copy.copy(Init_sol_P[i,:])
    P0[Bumps[2]]=0
    W0=Init_sol_W[i,:]
    O0=Init_sol_O[i,:]
    Sol=VegModelII_Riet_Spec_1D_02pi(L,N,M,tmax,dt,Dt,prec,P0+0*np.random.rand(N),W0+0*np.random.rand(N),O0+0*np.random.rand(N),param2)
    return(Sol)


results=Parallel(n_jobs=n_rains)(delayed(run_with_removed_patch)(Init_sol_P=Init_sol_P,Init_sol_W=Init_sol_W,Init_sol_O=Init_sol_O,r=r) for r in range(n_rains))


with open(path+"sol_n_%.d_Dw_%.4f_Dp_%.4f_Do_%.4f.txt"%(n_mode,f_Dw,f_Dp,f_Do), "wb") as fp:
    pickle.dump(results,fp)


with open(path+'Rains.txt',"wb") as fp:
    pickle.dump(rains,fp)
