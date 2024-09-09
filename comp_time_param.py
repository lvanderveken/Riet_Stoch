import pickle
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from static_1Dfunction import *
from IPython.display import display, HTML
from matplotlib import cm
import seaborn
import matplotlib
from matplotlib.gridspec import GridSpec

path='sensi_param/'
plt.rc('font', size=14) 


list_files=[
        "sol_n_4_Dw_0.1000_Dp_0.1000_Do_1.0000",
        "sol_n_4_Dw_0.2000_Dp_0.2000_Do_1.0000",
        "sol_n_4_Dw_0.3000_Dp_0.3000_Do_1.0000",
        "sol_n_4_Dw_0.4000_Dp_0.4000_Do_1.0000",
        "sol_n_4_Dw_0.5000_Dp_0.5000_Do_1.0000",
        "sol_n_4_Dw_0.6000_Dp_0.6000_Do_1.0000",
        "sol_n_4_Dw_0.7000_Dp_0.7000_Do_1.0000",
        "sol_n_4_Dw_0.8000_Dp_0.8000_Do_1.0000",
        "sol_n_4_Dw_0.9000_Dp_0.9000_Do_1.0000",
        "sol_n_4_Dw_1.0000_Dp_1.0000_Do_1.0000",
        ]

n_mode=[4,4,4,4,4,4]
f_Dw=np.arange(0.1,1.1,0.1)
f_Db=f_Dw
f_Do=[1]



  
dx=2*np.pi/100
L=100
x=np.arange(0,2*np.pi,dx)*(L/(2*np.pi))



with open(path+'Rains.txt','rb') as fp:
    rains=pickle.load(fp)

Max_time=np.zeros(len(list_files))

for n in range(len(list_files)):
    with open(path+list_files[n]+'.txt', "rb") as fp:   
        Sol = (pickle.load(fp))

    n_rains=np.shape(rains)[0]
    t=Sol[0][-1]
    index_t_stab=np.zeros(n_rains,dtype=int)
    for i in range(n_rains):
        j=10
        diff=100
        while diff>10**(-2):
            diff=np.max(np.abs(Sol[i][0][j+1,:]-Sol[i][0][j,:]))
            j=j+1
        index_t_stab[i]=int(j)
    Max_time[n]=np.max(t[index_t_stab])
    #print('f_Db=f_Dw=%.2f'%(f_Db[n]))
    #print('f_Do=%.2f'%(f_Do[n]))
    #print(Max_time[n])
    #print(index_t_stab)


plt.rc('font', size=50)
fig, ax = plt.subplots(1,1,figsize=(30,20))

ax.plot(f_Db,Max_time,marker='o',color='k')
ax.plot(np.linspace(0.1,1,1001),Max_time[0]/np.sqrt((1/f_Db[0]))*np.sqrt(1/np.linspace(0.1,1,1001)),color='r')
ax.set_xlabel('factor applied on the diffusion coefficient of biomass and soil water (f)',fontsize=40)
ax.set_ylabel('Rearregement time [d]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figure/time_param/time_diff_sqrt.png',bbox_inches='tight')

plt.rc('font', size=20)
fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.plot(f_Db,Max_time,marker='o',color='k')
ax.plot(np.linspace(0.1,1,1001),Max_time[0]/((1/f_Db[0]))*(1/np.linspace(0.1,1,1001)),color='r')
ax.set_xlabel('factor applied on the diffusion coefficient of biomass and soil water')
ax.set_ylabel('Rearregement time [d]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figure/time_param/time_diff_2.png')

fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.plot(f_Db,Max_time,marker='o',color='k')
ax.plot(np.linspace(0.1,1,1001),Max_time[0]/((1/f_Db[0]))**(3/4)*(1/np.linspace(0.1,1,1001))**(3/4),color='r')
ax.set_xlabel('factor applied on the diffusion coefficient of biomass and soil water')
ax.set_ylabel('Rearregement time [d]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figure/time_param/time_diff_3.png')


