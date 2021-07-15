# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:06:40 2021

@author: talebf
"""
import numpy as np
from numpy import pi
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os
import os.path
from matplotlib import cm
from scipy import signal 
from scipy import interpolate
from scipy.optimize import curve_fit
from numpy import exp, asarray
from scipy.optimize import curve_fit
from scipy.constants import c

um = 10**-6
THz = 10**12

doped_dir = r'E:\CURPROJECT\Conductivity\2021_05_14\p-doped GaAs_C 18817'
undoped_dir = r"E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25"

L=0.7*um # um film
d_P=530*um # um doped
d2=(530*um-L) # doped substrate - film
d=508*um # 'reference' substrate

os.chdir(doped_dir)

npzfile=np.load("signal_doped.npz") #,ref_TD,time_r,sig_TD,time_s
ref_TD_P=npzfile["arr_0"]
time_r_P=npzfile["arr_1"]
sig_TD_P=npzfile["arr_2"]
time_s_P=npzfile["arr_3"]

npzfile=np.load("FFT_doped.npz") #ref_FD,ref_Ph,freq_s,sig_FD,sig_Ph,freq_s
ref_FD_P=npzfile["arr_0"]
ref_Ph_p=npzfile["arr_1"]
freq_s_P_ref=npzfile["arr_2"]
sig_FD_P=npzfile["arr_3"]
sig_Ph_P=npzfile["arr_4"]

npzfile=np.load("Parameter_doped.npz") #n_mean,K_mean,A_mean, freq_s
n_mean_P=npzfile["arr_0"]
K_mean_P=npzfile["arr_1"]
A_mean_P=npzfile["arr_2"]
freq_s_P=npzfile["arr_3"]

## load undoped samples
os.chdir(undoped_dir)

npzfile=np.load("signal_undoped.npz") #,ref_TD,time_r,sig_TD,time_s
ref_TD_u=npzfile["arr_0"]
time_r_u=npzfile["arr_1"]
sig_TD_u=npzfile["arr_2"]
time_s_u=npzfile["arr_3"]

npzfile=np.load("FFT_undoped.npz") #ref_FD,freq_s,sig_FD,freq_s
ref_FD_u=npzfile["arr_0"]
ref_Ph_u=npzfile["arr_1"]
freq_s_u_ref=npzfile["arr_2"]
sig_FD_u=npzfile["arr_3"]
sig_Ph_u=npzfile["arr_4"]

npzfile=np.load("Parameter_undoped.npz") #n_mean,K_mean,A_mean, freq_s
n_mean_u=npzfile["arr_0"]
K_mean_u=npzfile["arr_1"]
A_mean_u=npzfile["arr_2"]
freq_s_u=npzfile["arr_3"]

indx= (freq_s_u>=0.1) & (freq_s_u<3)
indx3=(freq_s_u>= 0.11 ) & (freq_s_u<= 1)

Trans12= 2/(n_mean_u+1)
Trans21=(2*n_mean_u)/(n_mean_u+1)


# f1=np.exp(-2*np.pi*freq_s_u*d*K_mean_u/c)
# f2=np.exp(1j*2*np.pi*freq_s_u*d*(n_mean_u-1)/c)
# TranF=Trans12*Trans21*f1*f2

# calculate the transmission through matel film T23
n2_complex=np.zeros([len(time_s_u)], dtype=complex)
for i in range(len(n_mean_u)):
    n2_complex[i]=np.complex(n_mean_u[i],K_mean_u[i])
###
Trans12= 2/(n2_complex+1)
Trans21=(2*n2_complex)/(n2_complex+1) 
f=np.exp(1j*2*np.pi*freq_s_u*d*(n2_complex-1)/c)
TranF=Trans12*Trans21*f
   
T_undop=(sig_FD_u*np.exp(1j*sig_Ph_u))/(ref_FD_u*np.exp(1j*ref_Ph_u))
T_dop=(sig_FD_P*np.exp(1j*sig_Ph_P))/(ref_FD_P*np.exp(1j*ref_Ph_p))

########################################################################################################################
freq_s_u = freq_s_u[indx]
T_dop = T_dop[indx]
T_undop = T_undop[indx]

for idx in range(len(freq_s_u)):
    if idx % 50 != 0:
        continue


n_sub = n_mean_u[indx]

#K_mean_u = 0 # n_sub measured
k_sub = 0 # assuming substrate nonabsorbing

n1 = 1
omega = 2*pi*freq_s_u

n3 = n_sub + 1j*k_sub

n2 = 3.7 + 1j*100

t01, t12, t23 = 2*n1/(n1+n2), 2*n2/(n2+n3), 2*n3/(n3+n1)
t = t01*t12*t23

r01, r12 = (n1 - n2)/(n1+n2), (n2-n3)/(n2+n3)
r = r01*r12

T_model = t*exp(1j*n2*L*omega/c)*exp(1j*n3*d*omega/c)/(1+r*exp(2j*n2*L*omega/c))

"""
t01, t12, tr01 = 2*n1/(n1+n2), 2*n1/(n1+n2), 2*n1/(n1+n3)
r10, r12 = (n2-n1)/(n2+n1), (n2-n3)/(n2+n3)

t = t01*t12/tr01
r = r10*r12

T_model = t*exp(1j*(n2-n1)*L*omega/c)/(1-r*exp(2j*n2*L*omega/c))
"""

T_mess=T_dop/T_undop

print(T_model)
plt.plot(freq_s_u, T_model.real, label='T_model real')
plt.plot(freq_s_u, T_model.imag, label='T_model imag')

plt.plot(freq_s_u, T_mess.real, label='T_mess real')
plt.plot(freq_s_u, T_mess.imag, label='T_mess imag')
plt.legend()
plt.show()
