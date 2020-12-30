#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_16_statistic.csv')
df = df[~np.isnan(df['green_mean'])]
cells = list(set(df['channel']))
# df = df[df['channel'] == 'ch_0']

fig1, ax = plt.subplots(1, 1)
for cell in cells:
    ax.plot(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['green_mean'], '--g')
    ax.plot(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['red_mean'], '-r')
ax.set_ylim(100, 50000)
ax.set_yscale('log')
fig1.show()

fig1, ax = plt.subplots(1, 1)
for cell in cells:
    ax.plot(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
ax.set_xscale('log')
ax.set_xlim(100, 1000)
ax.set_yscale('log')
ax.set_ylim(100, 2000)
fig1.show()
