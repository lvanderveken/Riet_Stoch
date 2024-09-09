import numpy as np
#import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.linalg import toeplitz
from scipy.spatial import distance
from joblib import Parallel, delayed
import pickle
import os


def Spat_cov_matrix_periodic(x,lmb):
    #Create the square root of a covariance matrix of size N with a spatial structure e^(-x^2/lmb^2)
    D1=np.zeros((N,N))
    D2=np.zeros((N,N))

    for i in range(N):
        for j in range(i,N):
            D1[i,j]=np.exp(-((np.abs(x[i]-x[j]))/lmb)**2)
            D2[i,j]=np.exp(-(np.abs((L-np.abs((x[i]-x[j]))))/lmb)**2)


    DD=np.maximum(D1,D2)
    A=DD+DD.T-np.diagflat(np.ones(N))
    U,D,V=np.linalg.svd(A)
    D_sqrt=np.diagflat(np.sqrt(D))
    A_sqrt=np.matmul(U,np.matmul(D_sqrt,V))
    return(A_sqrt)


def create_noise(sigma,lmb_s,lmb_t,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)
        
    A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(N)*0.01
    for n in range(M-1):
        Sol[:,n+1]=Sol[:,n]+(-c*Sol[:,n])*dt+np.sqrt(dt)*sigma_e*A@np.random.randn(N)
    return(Sol[:,M_stab:])


def create_noise_lmb_t_zero(sigma,lmb_s,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)

    A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    #c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    #sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(N)*0.01
    for n in range(M):
        Sol[:,n]=sigma*A@np.random.randn(N)
    return(Sol[:,M_stab:])

def create_noise_lmb_s_zero(sigma,lmb_t,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)

    #A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(N)*0.01
    for n in range(M-1):
        Sol[:,n+1]=Sol[:,n]+(-c*Sol[:,n])*dt+np.sqrt(dt)*sigma_e*np.random.randn(N)
    return(Sol[:,M_stab:])

def create_noise_lmb_s_inf(sigma,lmb_t,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)

    #A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(1)*0.01*np.ones(N)
    for n in range(M-1):
        Sol[:,n+1]=Sol[:,n]+(-c*Sol[:,n])*dt+np.sqrt(dt)*sigma_e*np.random.randn(1)*np.ones(N)
    return(Sol[:,M_stab:])

def create_noise_lmb_t_zero_lmb_s_zero(sigma,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)

    #A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    #c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    #sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(N)*0.01
    for n in range(M):
        Sol[:,n]=sigma*np.random.randn(N)
    return(Sol[:,M_stab:])


def create_noise_lmb_t_zero_lmb_s_inf(sigma,tmax,dt,L,N):
    t_stab=1000
    #Time discretisation
    M=int((tmax+t_stab)/dt)
    t_simul=np.arange(0,tmax+t_stab,dt)
    M_stab=int(t_stab/dt)
    #Spatial discretisation
    dx=L/N
    x=np.arange(0,L,dx)

    #A=Spat_cov_matrix_periodic(x,lmb_s) #square root of the covariance matrix associated with the spatial noise

    #c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
    #sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process

    Sol=np.zeros((N,M))
    Sol[:,0]=np.random.randn(1)*0.01*np.ones(N)
    for n in range(M):
        Sol[:,n]=sigma*np.random.randn(1)*np.ones(N)
    return(Sol[:,M_stab:])


####################################################################

#Time discretisation
tmax=20000
dt=0.1
M=int(tmax/dt)
t=np.arange(0,tmax,dt)
#Spatial discretisation
L=100
N=100
dx=L/N
x=np.arange(0,L,dx)




path = 'noise_tmax_%.1f_dt_%.1f_L_%.1f'%(tmax,dt,L)


try:
    os.mkdir(path)
    print("Folder %s created!" % path)
except FileExistsError:
    print("Folder %s already exists" % path)






#Parameter of the noise
lmb_s=np.array([0,1,5,10,15,20,25,30,35,40,45,50,1000])
lmb_t=np.array([0,0.1,1,10,100,1000,10000])
sigma=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # Standard deviation of the final noise


n_real=15  # number of realisations





with open(path+"/Param.txt", "wb") as fp:
    pickle.dump([sigma,lmb_t,lmb_s], fp)


with open(path+"/axes.txt", "wb") as fp:
    pickle.dump([t,x], fp)





#Parameter of the noise
lmb_s=np.array([0,1,5,10,15,20,25,30,35,40,45,50,1000])
lmb_t=np.array([0,0.1,1,10,100,1000,10000])
#sigma=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # Standard deviation of the final noise
sigma=np.array([0.3])

n_lmb_s=np.shape(lmb_s)[0]
n_lmb_t=np.shape(lmb_t)[0]
n_sigma=np.shape(sigma)[0]


c=1/lmb_t #Parameter of the Ornstein-Ulhenbeck process
#sigma_e=sigma*np.sqrt(2*c) # Standard deviation of the noise applied to the Ornstein-Ulhenbeck process





for i in range(n_sigma):
    for j in range(1,n_lmb_t):
        for k in range(1,n_lmb_s):
            results =Parallel(n_jobs=n_real)(delayed(create_noise)(sigma[i],lmb_s=lmb_s[k],lmb_t=lmb_t[j],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
            print('Runs for the following parameters')
            print('sigma=%.1f'%(sigma[i]))
            print('lmb_t=%.1f'%(lmb_t[j]))
            print('lmb_s=%.1f'%(lmb_s[k]))
            print('---------------------------')
            print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_%.1f'%(sigma[i],lmb_t[j],lmb_s[k]))
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[i],lmb_t[j],lmb_s[k]), "wb") as fp:
                pickle.dump(results, fp)






for i in range(n_sigma):
    for k in range(1,n_lmb_s):
            results =Parallel(n_jobs=n_real)(delayed(create_noise_lmb_t_zero)(sigma[i],lmb_s=lmb_s[k],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
            print('Runs for the following parameters')
            print('sigma=%.1f'%(sigma[i]))
            print('lmb_t= %.1f'%(lmb_t[0]))
            print('lmb_s=%.1f'%(lmb_s[k]))
            print('---------------------------')
            print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_%.1f'%(sigma[i],lmb_t[0],lmb_s[k]))
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[i],lmb_t[0],lmb_s[k]), "wb") as fp:
                pickle.dump(results, fp)






for i in range(n_sigma):
    for k in range(1,n_lmb_t):
            results =Parallel(n_jobs=n_real)(delayed(create_noise_lmb_s_zero)(sigma[i],lmb_t=lmb_t[k],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
            print('Runs for the following parameters')
            print('sigma=%.1f'%(sigma[i]))
            print('lmb_t= %.1f'%(lmb_t[k]))
            print('lmb_s=%.1f'%(lmb_s[0]))
            print('---------------------------')
            print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_%.1f'%(sigma[i],lmb_t[k],lmb_s[0]))
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[i],lmb_t[k],lmb_s[0]), "wb") as fp:
                pickle.dump(results, fp)




for i in range(n_sigma):
    for k in range(1,n_lmb_t):
            results =Parallel(n_jobs=n_real)(delayed(create_noise_lmb_s_inf)(sigma[i],lmb_t=lmb_t[k],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
            print('Runs for the following parameters')
            print('sigma=%.1f'%(sigma[i]))
            print('lmb_t= %.1f'%(lmb_t[k]))
            print('lmb_s=inf')
            print('---------------------------')
            print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_inf'%(sigma[i],lmb_t[k]))
            with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma[i],lmb_t[k]), "wb") as fp:
                pickle.dump(results, fp)



for i in range(n_sigma):
    results =Parallel(n_jobs=n_real)(delayed(create_noise_lmb_t_zero_lmb_s_zero)(sigma[i],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
    print('Runs for the following parameters')
    print('sigma=%.1f'%(sigma[i]))
    print('lmb_t= %.1f'%(lmb_t[0]))
    print('lmb_s=%.1f'%(lmb_s[0]))
    print('---------------------------')
    print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_%.1f'%(sigma[i],lmb_t[0],lmb_s[0]))
    with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_%.1f_modif.txt"%(sigma[i],lmb_t[0],lmb_s[0]), "wb") as fp:
        pickle.dump(results, fp)





for i in range(n_sigma):
    results =Parallel(n_jobs=n_real)(delayed(create_noise_lmb_t_zero_lmb_s_inf)(sigma[i],tmax=tmax,dt=dt,L=L,N=N) for n in range(n_real))
    print('Runs for the following parameters')
    print('sigma=%.1f'%(sigma[i]))
    print('lmb_t= %.1f'%(lmb_t[0]))
    print('lmb_s=inf')
    print('---------------------------')
    print('Saving sigma= %.1f, lmb_t= %.1f_lmb_s_inf'%(sigma[i],lmb_t[0]))
    with open(path+"/Noise_sig_%.1f_lmb_t_%.1f_lmb_s_inf_modif.txt"%(sigma[i],lmb_t[0]), "wb") as fp:
        pickle.dump(results, fp)



