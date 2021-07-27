import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c

um = 10 ** -6
THz = 10 ** 12

#sample_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/GaAs Wafer Number 25/Sample/2021-05-14T12-48-40.306867-Sample_GaAs_undoped Wafer Number 25--X_6.000 mm-Y_15.000 mm.txt'
sample_path = r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Sample\2021-05-14T12-48-40.306867-Sample_GaAs_undoped Wafer Number 25--X_6.000 mm-Y_15.000 mm.txt'
sample_data = np.loadtxt(sample_path)

#ref_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/GaAs Wafer Number 25/Reference/2021-05-14T12-53-50.760769-Ref_GaAs_undopedWaferNumber25--X_-48.000mm-Y_15.000mm.txt'
ref_path = r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Reference\2021-05-14T12-53-50.760769-Ref_GaAs_undopedWaferNumber25--X_-48.000mm-Y_15.000mm.txt'
ref_data = np.loadtxt(ref_path)

sample_data[:, 1] -= sample_data[0, 1]
ref_data[:, 1] -= ref_data[0, 1]

sample_data[650:, 1] = 0
ref_data[650:, 1] = 0

"""
plt.plot(ref_data[:, 1], label='ref')
plt.plot(sample_data[:, 1], label='sample')
plt.title('TD')
plt.legend()
plt.show()
"""

f_max = 2.5*THz
f_min = 0.2*THz

sample_fft = np.fft.fft(sample_data[:, 1])
ref_fft = np.fft.fft(ref_data[:, 1])

freqs = THz*np.arange(0, len(sample_data), 1)/(sample_data[-1, 0]-sample_data[0, 0])
freq_range = (f_min < freqs) & (freqs < f_max)

sample_fft = sample_fft
ref_fft = ref_fft

T_mess = sample_fft/ref_fft

omega = 2*pi*freqs

d = 530 * um #+ 1*um

n = 1 - np.unwrap(np.angle(T_mess)) * c / (omega * d)

alpha = (-2 / d) * np.log(np.abs(T_mess) * (n + 1) ** 2 / (4 * n))

kappa = c*alpha/(2*omega)

#np.save('n_analytical', n)

n = n[freq_range]
alpha = alpha[freq_range]
kappa = kappa[freq_range]
freqs = freqs[freq_range]



fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].plot(freqs / 10 ** 12, n, label='n analytical')
ax[0].set_xlabel('Frequency (THz)')
ax[0].set_ylabel('Refractive index')
ax[0].legend()

ax[1].plot(freqs / 10 ** 12, alpha / 100, label='alpha analytical')
ax[1].set_xlabel('Frequency (THz)')
ax[1].set_ylabel('Alpha (1/cm)')
ax[1].legend()

ax[2].plot(freqs / 10 ** 12, kappa, label='kappa analytical')
ax[2].set_xlabel('Frequency (THz)')
ax[2].set_ylabel('Kappa')
ax[2].legend()

plt.show()
