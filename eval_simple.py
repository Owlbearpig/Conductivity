import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c

um = 10 ** -6
THz = 10 ** 12

sample_path = r'E:\CURPROJECT\Conductivity\2021_05_14\p-doped GaAs_C 18817\Sample\2021-05-14T11-25-28.123213-Sample_GaAs_C 18817--X_-3.000 mm-Y_14.000 mm.txt'
sample_data = np.loadtxt(sample_path)

ref_path = r'E:\CURPROJECT\Conductivity\2021_05_14\p-doped GaAs_C 18817\Reference\2021-05-14T10-49-49.145531-Reference_GaAs_C 18817--X_0.000 mm-Y_10.000 mm.txt'
ref_data = np.loadtxt(ref_path)

sample_data[:, 1] -= sample_data[0, 1]
ref_data[:, 1] -= ref_data[0, 1]

sample_data[660:, 1] = 0
ref_data[660:, 1] = 0

"""
plt.plot(ref_data[:, 1], label='ref')
plt.plot(sample_data[:, 1], label='sample')
plt.legend()
plt.show()
"""

sample_fft = np.fft.fft(sample_data[:, 1])
ref_fft = np.fft.fft(ref_data[:, 1])

freqs = THz*np.arange(0, len(sample_data), 1)/(sample_data[-1, 0]-sample_data[0, 0])
freq_range = np.where(freqs < 3*THz)

freqs = freqs[freq_range]

sample_fft = sample_fft[freq_range]
ref_fft = ref_fft[freq_range]

T_mess = sample_fft/ref_fft

omega = 2*pi*freqs
L, d = 0.7*um, 508*um
n1 = 1
n = 3.6

k = 15
n2 = n + 1j*k

t01, t12, t23 = 2/(1+n2), 2*n2/(n2+n), 2*n/(n+1)
t = t01*t12*t23

r01, r12 = (1 - n2)/(1+n2), (n2-n)/(n2+n)
r = r01*r12

del_1, del_2 = n2*L*omega/c, n*d*omega/c

T_ref = exp(1j*1*(d+L)*omega/c)

T_sample = t*(1/(exp(-1j*(del_1+del_2)) + r*exp(1j*(del_1-del_2))))

T_model = T_sample/T_ref

fig, ax = plt.subplots(nrows=1, ncols=3)

print(np.mean(T_model.real-T_mess.real))
print(np.mean(T_model.imag-T_mess.imag))

ax[0].plot(freqs/10**12, T_model.real, label='model real')
ax[0].plot(freqs/10**12, T_mess.real, label='T_mess real')
ax[0].legend()

ax[1].plot(freqs/10**12, T_model.imag, label='measurement imag')
ax[1].plot(freqs/10**12, T_mess.imag, label='measurement imag')
ax[1].legend()

ax[2].plot(freqs/10**12, np.abs(T_mess), label='|measurement|')
ax[2].plot(freqs/10**12, np.abs(T_model), label='|model|')
ax[2].legend()

plt.xlim((0, 3))
plt.show()
