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

um = 10 ** -6
THz = 10 ** 12

doped_dir = r'E:\CURPROJECT\Conductivity\2021_05_14\p-doped GaAs_C 18817'
undoped_dir = r"E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25"

d_film = 0.7 * um  # thin doped film
d_sub = 530 * um - d_film # 'sample' sub doped
d_sub_ref = 508 * um # 'reference' sub undoped

# doped data
os.chdir(doped_dir)

npzfile = np.load("signal_doped.npz")  # ,ref_TD,time_r,sig_TD,time_s
ref_TD_P = npzfile["arr_0"]
ref_time_P = npzfile["arr_1"]
ref_sig_TD_P = npzfile["arr_2"]
time_s_P = npzfile["arr_3"]

npzfile = np.load("FFT_doped.npz")  # ref_FD,ref_Ph,freq_s,sig_FD,sig_Ph,freq_s
ref_FD_P = npzfile["arr_0"]
ref_Ph_p = npzfile["arr_1"]
freq_s_P_ref = npzfile["arr_2"]*THz
sig_FD_P = npzfile["arr_3"]
sig_Ph_P = npzfile["arr_4"]

# not sure where this is from / what it is
npzfile = np.load("Parameter_doped.npz")  # n_mean,K_mean,A_mean, freq_s
n_mean_P = npzfile["arr_0"]
K_mean_P = npzfile["arr_1"]
A_mean_P = npzfile["arr_2"]
freq_s_P = npzfile["arr_3"]*THz

# undoped data
os.chdir(undoped_dir)

npzfile = np.load("signal_undoped.npz")  # ,ref_TD,time_r,sig_TD,time_s
ref_TD_u = npzfile["arr_0"]
time_r_u = npzfile["arr_1"]
sig_TD_u = npzfile["arr_2"]
time_s_u = npzfile["arr_3"]

npzfile = np.load("FFT_undoped.npz")  # ref_FD,freq_s,sig_FD,freq_s
ref_FD_u = npzfile["arr_0"]
ref_Ph_u = npzfile["arr_1"]
freq_s_u_ref = npzfile["arr_2"]*THz
sig_FD_u = npzfile["arr_3"]
sig_Ph_u = npzfile["arr_4"]

npzfile = np.load("Parameter_undoped.npz")  # n_mean,K_mean,A_mean, freq_s
n_mean_u = npzfile["arr_0"]
K_mean_u = npzfile["arr_1"]
A_mean_u = npzfile["arr_2"]
freq_s_u = npzfile["arr_3"]*THz

f_slice = (freq_s_u >= 0.1*THz) & (freq_s_u < 3*THz)
freqs = freq_s_u[f_slice]

omega = 2 * pi * freqs

T_undop = (sig_FD_u * np.exp(1j * sig_Ph_u)) / (ref_FD_u * np.exp(1j * ref_Ph_u))
T_undop = T_undop[f_slice]

T_dop = (sig_FD_P * np.exp(1j * sig_Ph_P)) / (ref_FD_P * np.exp(1j * ref_Ph_p))
T_dop = T_dop[f_slice]

T_mess = T_dop / T_undop # Transfer function. Measured, (Film + Substrate) / Substrate

n_sub_ref = n_mean_u[f_slice] + 1j*K_mean_u[f_slice]
n1 = 1  # n vac.
########################################################################################################################

def model_1(n_film):
    """
    Eq. (5) Ulatowski 2020.
    1. Assuming substrates same thickness.
    2. FP of substrate windowed out.
    3. Additional 'air phase term' (not 100% sure why)
    """

    n2, n3 = n_film, n_sub_ref
    t01, t12, tr01 = 2 * n1 / (n1 + n2), 2 * n1 / (n1 + n2), 2 * n1 / (n1 + n3)
    r10, r12 = (n2 - n1) / (n2 + n1), (n2 - n3) / (n2 + n3)

    t = t01 * t12 / tr01
    r = r10 * r12

    T_num = t * exp(1j * (n2 - n1) * d_film * omega / c)
    T_denom = (1 - r * exp(2j * n2 * d_film * omega / c))

    return T_num / T_denom

def model_2(n_film):
    # Not sure where this is from nor assumptions
    n2, n3 = n_film, n_sub_ref
    t01, t12, t23 = 2 * n1 / (n1 + n2), 2 * n2 / (n2 + n3), 2 * n3 / (n3 + n1)
    t = t01 * t12 * t23

    r01, r12 = (n1 - n2) / (n1 + n2), (n2 - n3) / (n2 + n3)
    r = r01 * r12

    T_num = t * exp(1j * n2 * d_film * omega / c) * exp(1j * n3 * d_sub_ref * omega / c)
    T_denom = 1 + r * exp(2j * n2 * d_film * omega / c)

    return T_num / T_denom

def model_3(n_film):
    n2, n3 = n_film, n_sub_ref

    t01, t12, tr01 = 2 * n1 / (n1 + n2), 2 * n1 / (n1 + n2), 2 * n1 / (n1 + n3)
    r10, r12 = (n2 - n1) / (n2 + n1), (n2 - n3) / (n2 + n3)

    t = t01 * t12 / tr01
    r = r10 * r12

    T_num = t * exp(1j * n3 * (d_sub-d_sub_ref) * omega / c) * exp(1j * (n2-n1) * d_film * omega / c)
    T_denom = (1 + r * exp(2j * n2 * d_film * omega / c))

    return T_num / T_denom


for idx in range(len(freq_s_u)):
    if idx % 50 != 0:
        continue

n_guess = 3.6 + 1j * 0.1  # film, unknown

T_model = model_3(n_guess)

plt.plot(freqs, T_model.real, label='T_model real')
plt.plot(freqs, T_model.imag, label='T_model imag')
#plt.plot(freqs, np.abs(T_model), label='|T_model|')

plt.plot(freqs, T_mess.real, label='T_mess real')
plt.plot(freqs, T_mess.imag, label='T_mess imag')
#plt.plot(freqs, np.abs(T_mess), label='|T_mess|')

plt.xlabel('Frequency (THz)')
plt.ylabel('Transfer function')
plt.legend()
plt.show()
