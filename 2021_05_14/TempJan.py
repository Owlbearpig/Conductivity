# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:22:07 2021

@author: Buntem
"""

from importing import import_tds
from preprocessing import offset, fft
import numpy as np
import matplotlib.pyplot as plt


t, a, list_name, _, _ = import_tds(r'\\dfs\physik\home\Buntem\Dokumente\Conductivity Measurements\2021_05_14\GaAs Wafer Number 25\Reference')
t_s, a_s, list_name, _, _ = import_tds(r'\\dfs\physik\home\Buntem\Dokumente\Conductivity Measurements\2021_05_14\GaAs Wafer Number 25\Sample')

t, a, list_name, _, _ = import_tds(r'\\dfs\physik\home\Buntem\Dokumente\Conductivity Measurements\2021_05_14\GaAs Wafer Number 25\Original')
t, a = offset(t, a, 3)

a_r = a[10:,:]
a_s = a[:10,:]
t = t[0]
idt = t > 1850
a_s[:, idt] = 0
a_r[:, idt] = 0

plt.figure()
plt.plot(t,np.mean(a_r,axis=0), label='Ref')
plt.plot(t,np.mean(a_s,axis=0), label='Sam')
plt.xlabel('Time (ps)')
plt.legend()
plt.show()

f, A_r = fft(t, a_r, windowing=False)
f, A_s = fft(t, a_s, windowing=False)
f = f[0]
idf  = f > 0
plt.figure()
plt.plot(f[idf], 20*np.log10(np.abs(A_r[0, idf])))
plt.show()

idf  = (f >= 0.3) & (f <= 2.5)
plt.figure()
plt.plot(f[idf], -20*np.log10(np.abs(A_s[0, idf])/np.abs(A_r[0, idf])))

plt.show()
# temp = A.T
