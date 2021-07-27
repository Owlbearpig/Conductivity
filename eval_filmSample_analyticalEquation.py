import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c

um = 10 ** -6
THz = 10 ** 12

#sample_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/p-doped GaAs_C 18817/Sample/2021-05-14T11-25-28.123213-Sample_GaAs_C 18817--X_-3.000 mm-Y_14.000 mm.txt'
sample_path = r'E:\CURPROJECT\Conductivity\2021_05_14\p-doped GaAs_C 18817\Sample\2021-05-14T11-25-28.123213-Sample_GaAs_C 18817--X_-3.000 mm-Y_14.000 mm.txt'
sample_data = np.loadtxt(sample_path)

#ref_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/p-doped GaAs_C 18817/Reference/2021-05-14T10-49-49.145531-Reference_GaAs_C 18817--X_0.000 mm-Y_10.000 mm.txt'
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

f_min, f_max = 0.25*THz, 2.5*THz
freq_range = (f_min < freqs) & (freqs < f_max)

T_mess = sample_fft/ref_fft

omega = 2*pi*freqs
L, d = 0.7*um, 508*um
n = np.load('n_analytical.npy')

alpha = (-2 / L) * np.log(np.abs(T_mess) * (n + 1) ** 2 / (4 * n))

kappa = c*alpha/(2*omega)

freqs, kappa, alpha = freqs[freq_range], kappa[freq_range], alpha[freq_range]

plt.plot(freqs, alpha/100, label='alpha analytical (1/cm)')
plt.legend()
plt.show()
