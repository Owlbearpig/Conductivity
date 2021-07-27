import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = r"E:\MEGA\AG\BFWaveplates\Data\Ceramic\Sample1_000deg_1825ps_0µm-2Grad_D=3000.csv"

data_frame = pd.read_csv(csv_file)

kappa_key = None
for key in data_frame.keys():
    if 'kappa' in key:
        kappa_key = key
        break

plt.plot(data_frame[kappa_key])
plt.show()
