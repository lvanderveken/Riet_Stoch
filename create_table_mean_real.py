import pickle
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from mpl_toolkits.mplot3d import Axes3D
from static_1Dfunction import *
from IPython.display import display, HTML
from matplotlib import cm
import seaborn
import matplotlib as matp
from matplotlib.gridspec import GridSpec



tmax=20000
dt=0.1
L=100
n_mode=3
rain=0.8




path="noise_tmax_%.1f_dt_%.1f_L_%.1f"%(tmax,dt,L)
path_veg="noise_tmax_%.1f_dt_%.1f_L_%.1f/realist/R_%.2f/n=%.d"%(tmax,dt,L,rain,n_mode)


with open("/Param.txt", "rb") as fp:   
    Param = pickle.load(fp)

with open("/axes.txt", "rb") as fp:   
    axes = pickle.load(fp)

sigma=Param[0] #sigma
sigma_O=0.1
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

n_real=15

print('tmax')
print(tmax)
print('L')
print(L)
print(sigma)
print('Rain')
print(rain)



Mean_Temp_mean_Spat_mean=np.zeros((n_lmb_t,n_lmb_s))


# For a given sigma
ind_sigma=0
# Diagnostic of the noise
ind_lmb_t_P=2
ind_lmb_s_O=-1

print('B: lmb_t=%.1f'%(lmb_t[ind_lmb_t_P]))
print('O: lmb_s=inf')
print('Sigma=%.1f'%(sigma[ind_sigma]))


for i in range(n_lmb_t):
    for j in range(n_lmb_s):
        file=path_veg+"/VegMod_P_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_O_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma[ind_sigma],lmb_t[ind_lmb_t_P],lmb_s[j],sigma_O,lmb_t[i])
        with open(file, "rb") as fp:   
            Sol = (pickle.load(fp))
        Temp_mean_Spat_mean=np.zeros(n_real)
        print('Solution loaded')
        print('B: lmb_t=%.1f , lmb_s=%.1f'%(lmb_t[ind_lmb_t_P],lmb_s[j]))
        print('O: lmb_t=%.1f , lmb_s=inf'%(lmb_t[i]))
        for k in range(n_real):
            Temp_mean_Spat_mean[k]=np.mean(np.mean(Sol[k][0],axis=1),axis=0)
        Mean_Temp_mean_Spat_mean[i,j]=np.mean(Temp_mean_Spat_mean)
 
        
        
plt.rc('font', size=30) 
fig, ax = plt.subplots(1,1,figsize=(20,10))
fig.suptitle('L=%.1f R=%.1f $\lambda_t$=%.1f for the biomass and $\lambda_s$=inf for the surface water ($\sigma_{B}$ =%.2f,$\sigma_{O}$ =%.2f)'%(L,rain,lmb_t[ind_lmb_t_P],sigma[ind_sigma],sigma_O),fontsize=20)

im1 = ax.imshow(Mean_Temp_mean_Spat_mean,cmap='OrRd')

plt.yticks(np.arange(n_lmb_t),labels=lmb_t)
plt.xticks(np.arange(n_lmb_s),labels=lmb_s)



ax.set_xticks(np.arange(n_lmb_s))
ax.set_xticklabels(lmb_s)
ax.set_yticks(np.arange(n_lmb_t))
ax.set_yticklabels(lmb_t)


ax.set_xlabel('$\lambda_s$')
ax.set_ylabel('$\lambda_t$',rotation=0)


for i in range(n_lmb_t):
    for j in range(n_lmb_s):
        if Mean_Temp_mean_Spat_mean[i, j] >= np.percentile(Mean_Temp_mean_Spat_mean, 80):
            color_s = 'white'
        else:
            color_s = 'k'
        text_mean = ax.text(j, i, '%.1f'%(Mean_Temp_mean_Spat_mean[i, j]),ha="center", va="center", color=color_s)


#ax.set_title('Temporal mean of the spatial mean')


plt.savefig('figure/realist/Table_mean_n_%d_R_%.2f_sigma_B_%.2f_sigma_O_%.2f_tmax_%.1f_modif.png'%(n_mode,rain,sigma[ind_sigma],sigma_O,tmax))    


