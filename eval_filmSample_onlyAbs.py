import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0

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
freq_range = np.where(freqs < f_max)

T_mess = sample_fft/ref_fft

omega = 2*pi*freqs
L, d = 0.7*um, 508*um
n = 3.508

def T_model(k=0.0):
    n2 = n + 1j*k

    t01, t12, t23 = 2/(1+n2), 2*n2/(n2+n), 2*n/(n+1)
    t = t01*t12*t23

    r01, r12 = (1 - n2)/(1+n2), (n2-n)/(n2+n)
    r = r01*r12

    del_1, del_2 = n2*L*omega/c, n*d*omega/c

    T_ref = exp(1j*(d+L)*omega/c)

    T_sample = t*(1/(exp(-1j*(del_1+del_2)) + r*exp(1j*(del_1-del_2))))

    return T_sample/T_ref

k_arr = np.linspace(0, 40, 1000)
#idx = 200 # 1 THz at idx=200
f = np.array([])
k_res = np.array([])

for idx in freq_range[0]:
    if idx % 10 != 0 or freqs[idx] < f_min:
        continue
    f = np.append(f, freqs[idx])

    diff = np.array([])
    for k in k_arr:
        diff = np.append(diff, (np.abs(T_mess) - np.abs(T_model(k)))[idx])

    k_abs_argmin = np.argmin(np.abs(diff))
    k_res = np.append(k_res, k_arr[k_abs_argmin])
    print(f'kappa={k_arr[k_abs_argmin]} best fit at {round(freqs[idx]/10**12, 3)} THz')
    """
    plt.title(f'Frequency: {round(freqs[idx]/10**12, 3)}')
    plt.plot(k_arr, diff, label='|measurement|-|model|')
    plt.legend()
    plt.show()
    """


plt.plot(f/10**12, k_res, label='k from abs_val compare')
plt.xlabel('Frequency (THz)')
plt.ylabel('kappa')
plt.legend()
plt.show()

Z0 = 376.730

eps = (n + 1j*k_res)**2

sigma = (eps-1)*epsilon_0*Z0/1j

plt.plot(f/10**12, sigma.real/100, label='sigma real (1/cm 1/ohm)')
plt.plot(f/10**12, sigma.imag/100, label='sigma imag (1/cm 1/ohm)')
plt.xlabel('Frequency (THz)')
plt.ylabel('Sigma')
plt.legend()
plt.show()
