#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load


#%%
df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_90_statistic.csv')
df = df[~np.isnan(df['green_mean'])]
cells = list(set(df['channel']))
# df = df[df['channel'] == 'ch_0']

fig1, ax = plt.subplots(1, 2, figsize=(18, 10))
for cell in cells:
    ax[0].plot(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['green_mean'], '--g')
    ax[0].plot(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['red_mean'], '-r')
ax[0].set_ylim(100, 50000)
ax[0].set_yscale('log')

for cell in cells:
    ax[1].scatter(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
ax[1].set_xscale('log')
ax[1].set_xlim(100, 2000)
ax[1].set_yscale('log')
ax[1].set_ylim(100, 10000)

fig1.show()


#%%
df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_90_statistic.csv')
cells = list(set(df['channel']))
fig2, ax = plt.subplots(1, 1, figsize=(18, 10))
for cell in cells:
    ax.scatter(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['area'])

# ax.set_yscale('log')
ax.set_xlim(df['time_s'].min() + np.ptp(df['time_s'])*0.0,
            df['time_s'].min() + np.ptp(df['time_s'])*1.0)
fig2.show()
# fig1, ax = plt.subplots(1, 1)
# for cell in cells:
#     ax.plot(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
# ax.set_xscale('log')
# ax.set_xlim(100, 2000)
# ax.set_yscale('log')
# ax.set_ylim(100, 10000)
# fig1.show()
