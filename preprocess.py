import numpy as np
import matplotlib.pyplot as plt

ref = np.loadtxt(r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Reference\2021-05-14T12-53-50.760769-Ref_GaAs_undopedWaferNumber25--X_-48.000mm-Y_15.000mm.txt')
sample = np.loadtxt(r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Sample\2021-05-14T12-48-40.306867-Sample_GaAs_undoped Wafer Number 25--X_6.000 mm-Y_15.000 mm.txt')

ref = ref - ref[0, 1]
sample = sample - sample[0, 1]

np.savetxt(r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Reference\2021-05-14T12-53-50.760769-undoped-no_offset.txt', ref)
np.savetxt(r'E:\CURPROJECT\Conductivity\2021_05_14\GaAs Wafer Number 25\Sample\2021-05-14T12-48-40.306867-undoped-no_offset.txt', sample)


