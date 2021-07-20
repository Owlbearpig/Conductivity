import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c

um = 10 ** -6
THz = 10 ** 12

sample_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/GaAs Wafer Number 25/Sample/2021-05-14T12-48-40.306867-Sample_GaAs_undoped Wafer Number 25--X_6.000 mm-Y_15.000 mm.txt'
sample_data = np.loadtxt(sample_path)

ref_path = r'/media/alex/sda2/ProjectsOverflow/Conductivity/2021_05_14/GaAs Wafer Number 25/Reference/2021-05-14T12-53-50.760769-Ref_GaAs_undopedWaferNumber25--X_-48.000mm-Y_15.000mm.txt'
ref_data = np.loadtxt(ref_path)

sample_data[:, 1] -= sample_data[0, 1]
ref_data[:, 1] -= ref_data[0, 1]

sample_data[650:, 1] = 0
ref_data[650:, 1] = 0

"""
plt.plot(ref_data[:, 1], label='ref')
plt.plot(sample_data[:, 1], label='sample')
plt.legend()
plt.show()
"""

f_max = 2.5*THz

sample_fft = np.fft.fft(sample_data[:, 1])
ref_fft = np.fft.fft(ref_data[:, 1])

freqs = THz*np.arange(0, len(sample_data), 1)/(sample_data[-1, 0]-sample_data[0, 0])
freq_range = np.where(freqs < f_max)

freqs = freqs[freq_range]

sample_fft = sample_fft[freq_range]
ref_fft = ref_fft[freq_range]

T_mess = sample_fft/ref_fft

omega = 2*pi*freqs

d = 530 * um
n1 = 1


def T_model(n=3.6):

    n3 = n

    t01, t12 = 2/(n1+n3), 2*n3/(n1+n3)
    t = t01*t12

    del3 = n3*d*omega/c

    T_ref = exp(1j*1*d*omega/c)

    T_sample = t*exp(1j*del3)

    return T_sample/T_ref

idx = 155 # 1 THz at idx=200

n_arr = np.linspace(3.2, 3.9, 10000)

diff = np.array([])
for n in n_arr:
    diff = np.append(diff, (T_mess - T_model(n))[idx])

n_real_argmin = np.argmin(np.abs(diff.real))
n_imag_argmin = np.argmin(np.abs(diff.imag))
n_abs_argmin = np.argmin(np.abs(diff))

print(n_arr[n_real_argmin], 'n best fit real')
print(n_arr[n_imag_argmin], 'n best fit imag')
print(n_arr[n_abs_argmin], 'n best fit abs')

fig, ax = plt.subplots(nrows=1, ncols=4)
fig.suptitle(f'Frequency: {round(freqs[idx]/10**12, 3)}')

ax[0].plot(n_arr, diff.real, label='mess-model real')
ax[0].legend()

ax[1].plot(n_arr, diff.imag, label='mess-model imag')
ax[1].legend()

ax[2].plot(n_arr, np.abs(diff), label='|measurement|-|model|')
ax[2].legend()

ax[3].plot(n_arr, diff.real, label='mess-model real')
ax[3].plot(n_arr, diff.imag, label='mess-model imag')
ax[3].plot(n_arr, np.abs(diff), label='|measurement|-|model|')
ax[3].legend()

plt.show()

exit()

fig, ax = plt.subplots(nrows=1, ncols=3)

print(np.mean(T_model.real-T_mess.real), 'real mean diff')
print(np.mean(T_model.imag-T_mess.imag), 'imag mean diff')

ax[0].plot(freqs/10**12, T_model.real, label='model real')
ax[0].plot(freqs/10**12, T_mess.real, label='T_mess real')
ax[0].legend()

ax[1].plot(freqs/10**12, T_model.imag, label='model imag')
ax[1].plot(freqs/10**12, T_mess.imag, label='measurement imag')
ax[1].legend()

ax[2].plot(freqs/10**12, np.abs(T_mess), label='|measurement|')
ax[2].plot(freqs/10**12, np.abs(T_model), label='|model|')
ax[2].legend()

plt.xlim((0, f_max/10**12))
plt.show()
