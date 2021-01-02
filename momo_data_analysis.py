#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os
from tqdm import tqdm
import sys
sys.path.append(r'D:\python_code\data_explore')
import sciplot as splt
import seaborn as sns
splt.whitegrid()


def convert_time(time):
    h, s = divmod(time, 60*60)
    m, s = divmod(s, 60)
    h = h + m/60 + s/3600
    return h


#%%
dir = r'X:/chupan/mother machine/20201225_NCM_pECJ3_M5_L3/'

all_scv = [file for file in os.listdir(dir) if file.split('.')[-1] == 'csv']
dfs = [pd.read_csv(os.path.join(dir, ps)) for ps in tqdm(all_scv)]

for i, df in enumerate(dfs):
    chamber = [f'fov{i}_{ch}' for ch in df['channel']]
    df['chamber'] = chamber
dfs = pd.concat(dfs)
dfs.index = pd.Index(range(len(dfs)))
dfs['time_h'] = [convert_time(s) for s in dfs['time_s']-min(dfs['time_s'])]

fd_dfs = dfs[dfs['area'] > 100]
print(f'''all chambers {len(list(set(fd_dfs['chamber'])))}''')
#%%
fig1, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.hist(np.log(fd_dfs['area']), bins=100)
ax.set_xlabel('Cell area (log(pixels))')
ax.set_ylabel('Number')
fig1.show()
#%%

fig1, ax = plt.subplots(1, 2, figsize=(21*2, 10*2))
cells = list(set(fd_dfs['chamber']))
cells = np.random.choice(cells, 3)
fig2, ax = plt.subplots(1, 1, figsize=(18, 10))
for cell in cells:
    ax.plot(fd_dfs[fd_dfs['chamber'] == cell]['time_h'],
               fd_dfs[fd_dfs['chamber'] == cell]['area'])
    ax.scatter(fd_dfs[fd_dfs['chamber'] == cell]['time_h'],
               fd_dfs[fd_dfs['chamber'] == cell]['area'],
               s=158)

ax.set_xlim(fd_dfs['time_h'].min() + np.ptp(fd_dfs['time_h'])*0.0,
            fd_dfs['time_h'].min() + np.ptp(fd_dfs['time_h'])*0.2)
fig2.show()

#%%
ff_dfs = fd_dfs[~np.isnan(fd_dfs['green_mean'])]
fig1, ax = plt.subplots(1, 2, figsize=(21*2, 10*2))
cells = list(set(ff_dfs['chamber']))
cells = np.random.choice(cells, 50)
for cell in cells:
    ax[0].plot(ff_dfs[ff_dfs['chamber'] == cell]['time_h'],
               ff_dfs[ff_dfs['chamber'] == cell]['green_mean']/ff_dfs[ff_dfs['chamber'] == cell]['red_mean'], '--y')
    # ax[0].plot(ff_dfs[ff_dfs['chamber'] == cell]['time_h'], ff_dfs[ff_dfs['chamber'] == cell]['red_mean'], '-r')
# ax[0].set_ylim(100, 50000)
ax[0].set_yscale('log')
ax[0].set_xlim(0)
ax[0].set_xlabel('Time (h)')
ax[0].set_ylabel('Fold increase (gfp/rfp)')

for cell in cells:
    ax[1].scatter(ff_dfs[ff_dfs['chamber'] == cell]['red_mean'], ff_dfs[ff_dfs['chamber'] == cell]['green_mean'])
ax[1].set_xscale('log')
ax[1].set_xlim(100, 2000)
ax[1].set_yscale('log')
ax[1].set_ylim(100, 10000)
ax[1].set_xlabel('Rfp intensity')
ax[1].set_ylabel('Gfp intensity')
fig1.show()
#%%
cells = np.array(list(set(ff_dfs['chamber'])))
cell_ints = [True if ff_dfs[ff_dfs['chamber'] == cell]['time_h'].max() > 40 else False for cell in cells]
fc_list = []
for cell in cells[cell_ints]:
    fc_i = ff_dfs[ff_dfs['chamber'] == cell].iloc[0]['green_mean'] / ff_dfs[ff_dfs['chamber'] == cell].iloc[0]['red_mean']
    fc_f = ff_dfs[ff_dfs['chamber'] == cell].iloc[-1]['green_mean'] / ff_dfs[ff_dfs['chamber'] == cell].iloc[-1]['red_mean']
    fc = fc_i / fc_f
    fc_list.append(fc)

fc = pd.DataFrame(data=dict(chamber=cells[cell_ints],
                            fold_change=fc_list))

fig3, ax = plt.subplots(1, 1)
sns.histplot(data=fc, x='fold_change', bins=100, log_scale=True, ax=ax)

fig3.show()

#%%
fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.histplot(x=ff_dfs['red_mean'], y=ff_dfs['green_mean'],
             log_scale=(True, True),
             cbar=True,
             cbar_kws=dict(shrink=.75),
             pthresh=.1,
             ax=ax)
ax.set_xlabel('Rfp intensity (a.u.)')
ax.set_ylabel('Gfp intensity (a.u.)')

fig2.show()

#%%
df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_55_statistic.csv')
df = df[~np.isnan(df['green_mean'])]
cells = list(set(df['channel']))
# df = df[df['channel'] == 'ch_0']

fig1, ax = plt.subplots(1, 2, figsize=(18, 10))
for cell in cells:
    ax[0].plot(df[df['channel'] == cell]['time_s'],
               df[df['channel'] == cell]['green_mean']/df[df['channel'] == cell]['red_mean'], '--y')
    # ax[0].plot(df[df['channel'] == cell]['time_s'], df[df['channel'] == cell]['red_mean'], '-r')
# ax[0].set_ylim(100, 50000)
ax[0].set_yscale('log')

for cell in cells:
    ax[1].scatter(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
ax[1].set_xscale('log')
ax[1].set_xlim(100, 2000)
ax[1].set_yscale('log')
ax[1].set_ylim(100, 10000)

fig1.show()


#%%
# df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_55_statistic.csv')
cells = list(set(dfs['chamber']))
fig2, ax = plt.subplots(1, 1, figsize=(18, 10))
for cell in cells:
    ax.plot(dfs[dfs['channel'] == cell]['time_h'], dfs[dfs['channel'] == cell]['area'])

# ax.set_yscale('log')
ax.set_xlim(dfs['time_h'].min() + np.ptp(dfs['time_h'])*0.0,
            dfs['time_h'].min() + np.ptp(dfs['time_h'])*1)
fig2.show()
# fig1, ax = plt.subplots(1, 1)
# for cell in cells:
#     ax.plot(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
# ax.set_xscale('log')
# ax.set_xlim(100, 2000)
# ax.set_yscale('log')
# ax.set_ylim(100, 10000)
# fig1.show()
#%%
db = load(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_1.jl')
fig2, ax = plt.subplots(1,1)
ax.imshow(db['chamber_cells_mask']['ch_1'][5])

fig2.show()