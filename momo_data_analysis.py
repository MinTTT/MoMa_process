# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import sys
from joblib import delayed, Parallel, load
import seaborn as sns

# from dask.diagnostics import ProgressBar
# import dask
# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=4, threads_per_worker=8)
# client = Client(cluster)

sys.path.append(r'D:\python_code\data_explore')
try:
    import sciplot as splt

    splt.whitegrid()
except ModuleNotFoundError:
    print('''Module sciplot wasn't founded.''')


def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


def get_growth_rate(cell_df, **kwargs):
    """
    filter cell's growth cycle data, compute instantaneous growth rate.
    :param cell_df: pandas data frame
    :param kwargs: 'sizetype', length or area
    :return: ndarray, (2, len of dataframe)
    """

    if 'sizetype' in kwargs:
        st = kwargs['sizetype']
    else:
        st = 'length'
    if cell_df['time_h'].max() < 75:
        return None
    cell_area = cell_df[st]
    area_dd = np.diff(np.diff(cell_area))
    outlier = np.where(1 * cell_area[0:-2] < -area_dd)[0] + 1
    mask = np.array([True] * len(cell_area))
    mask[outlier] = False
    fild_cell_df = cell_df[mask]
    area_dd = np.diff(np.diff(fild_cell_df[st]))
    outlier2 = np.where(1 * fild_cell_df[st][0:-2] < area_dd)[0] + 1
    mask = np.array([True] * len(fild_cell_df))
    mask[outlier2] = False
    fild_cell_df = fild_cell_df[mask]  # filtered data, excluding outlier
    fild_area_diff = np.diff(fild_cell_df[st])
    div_thre = 0.8
    min_p = fild_cell_df[st] * (1 - div_thre)
    max_p = fild_cell_df[st] * div_thre
    filter_peaks = np.where(np.logical_and(-fild_area_diff > min_p[:-1], -fild_area_diff < max_p[:-1]))[
        0]  # division peaks
    peak_num = len(filter_peaks)
    peaks_sclice = [0] + list(filter_peaks + 1) + [len(fild_cell_df)]
    num_peaks = np.diff(peaks_sclice)
    len_index = np.where(num_peaks >= 4)[0]  # one cell cycle must have more than 4 records.
    div_slice_all = [slice(peaks_sclice[i], peaks_sclice[i + 1]) for i in range(peak_num + 1)]
    div_slice = [div_slice_all[i] for i in len_index]  # cell cycle slice
    if len(div_slice) < 4:  # at least divide 4 times.
        return None
    time_all = []
    rate_all = []
    for sl in div_slice:
        length = fild_cell_df[st].iloc[sl]
        time = fild_cell_df['time_h'].iloc[sl]
        time_diff = np.diff(time)
        rate = np.diff(length) / time_diff
        time = time[:-1] + time_diff / 2
        rate_all.append(rate)
        time_all.append(time)

    rate_all = np.concatenate(rate_all)
    time_all = np.concatenate(time_all)

    return np.array([time_all, rate_all]).T



def binned_average(mat, num=100, std=True, classify_only=False):
    """
    get binned average and standard

    :param mat:
    :param num:
    :return:
    """
    binned_num = num
    binned_time = mat[:, 0]
    # gr = high_imdata
    bins = np.linspace(binned_time.min(), binned_time.max(), num=binned_num, endpoint=False)
    index = np.searchsorted(bins, binned_time, side='right')
    # binned_avg = np.array(Parallel(n_jobs=8, backend='threading')(delayed(np.average)(mat[index == binned_i + 1, :], axis=0)
    #                                           for binned_i in range(binned_num)))
    # binned_std = np.array(Parallel(n_jobs=8, backend='threading')(delayed(np.std)(mat[index == binned_i + 1, :], axis=0)
    #                                           for binned_i in range(binned_num)))
    binned_avg = np.array([np.average(mat[index == binned_i + 1, :], axis=0) for binned_i in range(binned_num)])
    if classify_only:
        ref_avg = np.zeros(len(binned_time))
        for binned_i in range(binned_num):
            ref_avg[index == binned_i + 1] = binned_avg[binned_i, 0]
        mat[:, 0] = ref_avg
        return mat

    if std:
        binned_std = np.array([np.std(mat[index == binned_i + 1, :], axis=0) for binned_i in range(binned_num)])
        return [binned_avg, binned_std]
    return binned_avg


# @dask.delayed
def daskdf_parse(df, keys, na):
    return df[df[keys] == na]


# %% load statistic data
ps = r'./test_data_set/csv_data/mothers_raw_dic.jl'
# cells_df = cells_df.compute()
cells_df = load(ps)
cells_name = list(cells_df.keys())
# %% filter data
print('calculate instantaneous growth.')
cells_growth_rate = Parallel(n_jobs=64, backend='threading')(
    delayed(get_growth_rate)(cells_df[cn]) for cn in tqdm(cells_name))
cells_growth_rate = {cn: cells_growth_rate[i] for i, cn in tqdm(enumerate(cells_name))}

cells_growth_rate = {cn: cells_growth_rate[cn] for cn in cells_name
                     if isinstance(cells_growth_rate[cn], np.ndarray)}
gr_array = [cells_growth_rate[na] for na in tqdm(list(cells_growth_rate.keys())) if
            isinstance(cells_growth_rate[na], np.ndarray)]
gr_array = np.vstack(gr_array)
# binned average
bin_num = 250
time = gr_array[:, 0]
gr = gr_array[:, 1]
avg, std = binned_average(gr_array, bin_num)
time_avg = avg[:, 0]
gr_avg = avg[:, 1]
gr_std = std[:, 1]
std_up = gr_avg + gr_std
std_down = gr_avg - gr_std

cells = np.random.choice(list(cells_growth_rate.keys()), 20)

fig1, ax = plt.subplots(1, 1)
for na in tqdm(cells):
    data = cells_growth_rate[na]
    if isinstance(data, np.ndarray):
        ax.plot(data[:, 0], data[:, 1], color='#ABB2B9', lw=0.5, alpha=0.4)
ax.plot(data[:, 0], data[:, 1], color='#E67E22', lw=1.2, alpha=0.5)
ax.scatter(time_avg, gr_avg, s=40, color='#3498DB')
ax.plot(time_avg, std_up, '--', lw=3, color='#5DADE2')
ax.plot(time_avg, std_down, '--', lw=3, color='#5DADE2')
ax.set_xlim(0, time.max())
ax.set_ylim(-80, 220)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Growth rate (1/h)')
fig1.show()
# %% binned 4
print("classify growth type.")
cells_name = list(cells_growth_rate.keys())
four_binned = Parallel(n_jobs=1000, backend='threading')(
    delayed(list)(binned_average(cells_growth_rate[cn], num=4, std=False)[:, 1])
    for cn in tqdm(cells_name))
# four_binned = [list(binned_average(cells_growth_rate[cn][:, 0], cells_growth_rate[cn][:, 1], num=4)[1])
#                for cn in cells_name]
four_binned = np.array(four_binned)

# K-MES classify
from sklearn.cluster import KMeans

km_model = KMeans(n_clusters=3, random_state=4).fit(four_binned)
km_label = km_model.labels_
colors_4 = ['#DBB38F', '#91DB8F', '#8FB7DB', '#D98FDB']  # brown, green, blue, violet
fig2, ax2 = plt.subplots(1, 1)
for label in range(4):
    color = colors_4[label]
    masked_data = four_binned[km_label == label, :]
    ax2.scatter(masked_data[:, 0], masked_data[:, -1], color=color)
ax2.set_xlabel('Growth rate (before shift)')
ax2.set_ylabel('Growth rate (after shift)')
fig2.show()
clfd_cells = []
for i in range(3):
    clfd_chamber = np.where(km_label == i)[0]
    clfd_cells.append([cells_name[ind] for ind in clfd_chamber])

# plot classified
fig3, axes = plt.subplots(3, 1, figsize=(15, 15))
for i, ax in enumerate(axes):
    clfd_chamber = np.where(km_label == i)[0]
    clfd_chamber = [cells_name[ind] for ind in clfd_chamber]
    gr_array = [cells_growth_rate[na] for na in tqdm(clfd_chamber) if
                isinstance(cells_growth_rate[na], np.ndarray)]
    gr_array = np.vstack(gr_array)
    # binned average
    bin_num = 250
    _avg, _std = binned_average(gr_array, bin_num)
    time_avg, gr_avg = _avg[:, 0], _avg[:, 1]
    time_std, gr_std = _std[:, 0], _std[:, 1]

    std_up = gr_avg + gr_std
    std_down = gr_avg - gr_std
    cells = np.random.choice(clfd_chamber, 10)
    for na in tqdm(cells):
        data = cells_growth_rate[na]
        if isinstance(data, np.ndarray):
            ax.plot(data[:, 0], data[:, 1], color='#ABB2B9', lw=0.5, alpha=0.4)  # show in grey
    ax.plot(data[:, 0], data[:, 1], color='#E67E22', lw=1.2, alpha=0.5)
    ax.plot(time_avg, std_up, '--', lw=3, color='#5DADE2')
    ax.plot(time_avg, std_down, '--', lw=3, color='#5DADE2')
    ax.scatter(time_avg, gr_avg, s=40, color='#3498DB')
    ax.set_xlim(0, time.max())
    ax.set_ylim(-80, 220)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Growth rate (1/h)')
fig3.show()

# %%
fluor_binnum = 100
clfd_cell = clfd_cells[0]
cells_green = []
cells_red = []
cells_time = []
flu_cells_dict = {cn: None for cn in clfd_cell}
grey_color = dict(color='#ABB2B9', lw=0.5, alpha=0.5)
orange_color = dict(color='#E67E22', lw=2, alpha=0.6)
for cell in tqdm(clfd_cell):
    mask_df = cells_df[cell][~np.isnan(cells_df[cell]['green_medium'])]
    flu_cells_dict[cell] = mask_df
    cells_green.append(mask_df[['green_medium', 'red_medium', 'time_h']].values.reshape(-1, 3))
    # cells_red.append(mask_df['red_medium'].values.reshape(-1, 1))
    # cells_time.append(mask_df['time_h'].values.reshape(-1, 1))
cells_green = np.vstack(cells_green)
# cells_red = np.vstack(cells_red)
# cells_time = np.vstack(cells_time)

mean_flu, std = binned_average(cells_green[:, -1::-1], num=fluor_binnum)
bred_mean = mean_flu[:, 1]
bgreen_mean = mean_flu[:, -1]
# _, bred_mean, bred_std = binned_average(cells_time, cells_red, num=fluor_binnum)

fig4, ax = plt.subplots(1, 1)
sld_cells = np.random.choice(clfd_cell, 4)
for cell in tqdm(sld_cells):
    cell_data = cells_df[cell]
    mask_df = cells_df[cell][~np.isnan(cells_df[cell]['green_medium'])]
    ax.plot(mask_df['red_medium'], mask_df['green_medium'], color='#ABB2B9', lw=0.5, alpha=0.5)
ax.plot(mask_df['red_medium'], mask_df['green_medium'], color='#E67E22', lw=2, alpha=0.6)
ax.plot(bred_mean, bgreen_mean, lw=2, color='#3498DB')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('GFP intensity (a.u.)')
ax.set_xlabel('mCherry intensity (a.u.)')
ax.set_ylim(2, 3000)
ax.set_xlim(2, 6000)
fig4.show()

# %%
bin_num = 11
binned_fold_change = np.zeros((len(clfd_cell), bin_num))
binned_green = np.zeros((len(clfd_cell), bin_num))
binned_red = np.zeros((len(clfd_cell), bin_num))
binned_total_flu = np.zeros((len(clfd_cell), bin_num))

def flu_binned(index, cn):
    # g_max, r_max = flu_cells_dict[cn]['green_medium'].max(), flu_cells_dict[cn]['red_medium'].max()
    green_over_red = (flu_cells_dict[cn]['green_medium'] + 1) / (flu_cells_dict[cn]['red_medium'] + 1)
    green_over_red = green_over_red.values.reshape(-1, 1)
    total_flu = flu_cells_dict[cn]['green_medium'] + flu_cells_dict[cn]['red_medium']
    data = np.hstack([flu_cells_dict[cn][['time_h', 'green_medium', 'red_medium']].values, 
                      green_over_red,
                      total_flu.values.reshape(-1, 1)])
    binned = binned_average(data, bin_num, std=False)
    binned_fold_change[index, :] = binned[:, 3]
    binned_green[index, :] = binned[:, 1]
    binned_red[index, :] = binned[:, 2]
    binned_total_flu[index, :] = binned[:, 4]

_ = Parallel(n_jobs=32, backend='threading')(delayed(flu_binned)(index, cn) for index, cn in enumerate(tqdm(clfd_cell)))

fig5, ax5 = plt.subplots(1, 1)
# ax8.scatter(binned_red[:, -1], binned_green[:, -1])
sns.histplot(x=binned_red[:, -1], y=binned_green[:, -1], ax=ax5)
fig5.show()

green_mean, green_std = binned_green[:, -1].mean(), binned_green[:, -1].std()
red_mean, red_std = binned_red[:, -1].mean(), binned_red[:, -1].std()

mask_otler = np.where(np.logical_and(binned_green[:, -1] <= 1.5 * green_std + green_mean,
                               binned_red[:, -1] <= 1.0 * red_std + red_mean), True, False)
sub_clf_cells = [clfd_cell[i] for i in np.arange(len(clfd_cell))[mask_otler]]

fig6, ax6 = plt.subplots(1, 1)
# ax8.scatter(binned_red[:, -1], binned_green[:, -1])
sns.histplot(x=binned_red[:, -1][mask_otler], y=binned_green[:, -1][mask_otler], ax=ax6)
fig6.show()

log_bin_flu = np.log(binned_fold_change)

sub_log_bin_flu = log_bin_flu[mask_otler, :]

# %%
print('Modeling Fitting.')
km_model2 = KMeans(2, random_state=3).fit(sub_log_bin_flu[:, -3:-1])
km_label2 = km_model2.labels_

fig7, ax = plt.subplots(1, 1)
for i in list(set(km_label2)):
    ax.scatter(binned_fold_change[mask_otler][km_label2 == i, :][:, -2],
               binned_fold_change[mask_otler][km_label2 == i, :][:, -1])

ax.set_xlim(1e-3, 30)
ax.set_ylim(1e-2, 35)
ax.set_xscale('log')
ax.set_yscale('log')
fig7.show()

sub_clf_cells_name = {}
fig8, ax = plt.subplots(1, 1)
for label in list(set(km_label2)):
    mask = km_label2 == label
    selected_cell = [sub_clf_cells[i] for i in np.arange(len(km_label2))[mask]]
    sub_clf_cells_name[label] = selected_cell
    cells = np.random.choice(selected_cell, 20)
    for cell in cells:
        ax.plot(flu_cells_dict[cell]['red_medium'], flu_cells_dict[cell]['green_medium'], color=colors_4[label])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('GFP intensity (a.u.)')
ax.set_xlabel('mCherry intensity (a.u.)')
ax.set_ylim(2, 3000)
ax.set_xlim(2, 6000)
fig8.show()

sub_clf_data = {}
sub_clf_bin_mean_data = {}
sub_clf_bin_std_data = {}

num_ax = len(list(set(km_label2)))
fig9, ax = plt.subplots(1, num_ax, figsize=(10 * num_ax, 8))
for label in list(set(km_label2)):
    selected_cell = sub_clf_cells_name[label]
    cells = np.random.choice(selected_cell, 20)
    for cell in cells:
        ax[label].plot(flu_cells_dict[cell]['red_medium'], flu_cells_dict[cell]['green_medium'], **grey_color)
    ax[label].plot(flu_cells_dict[cell]['red_medium'], flu_cells_dict[cell]['green_medium'], **orange_color)
    data = np.vstack([flu_cells_dict[cn][['time_h', 'green_medium', 'red_medium']].values for cn in selected_cell])
    sub_clf_data[label] = data
    binned_avg, binned_std = binned_average(data, 50)
    sub_clf_bin_std_data[label] = binned_std
    sub_clf_bin_mean_data[label] = binned_avg
    ax[label].plot(binned_avg[:, 2], binned_avg[:, 1], colors_4[label])
    ax[label].set_xscale('log')
    ax[label].set_yscale('log')
    ax[label].set_ylabel('GFP intensity (a.u.)')
    ax[label].set_xlabel('mCherry intensity (a.u.)')
    ax[label].set_ylim(2, 3000)
    ax[label].set_xlim(2, 6000)
fig9.show()

fig10, axes9 = plt.subplots(1, 2, figsize=(12*2, 10))
for i, ax10 in enumerate(axes9):
    binned_avg = sub_clf_bin_mean_data[i]
    ax10.plot(binned_avg[:, 0], binned_avg[:, 1], 'g')
    ax10.plot(binned_avg[:, 0], binned_avg[:, 2], 'r')
    ax10.errorbar(binned_avg[:, 0], binned_avg[:, 1], binned_std[:, 1])
    ax10.errorbar(binned_avg[:, 0], binned_avg[:, 2], binned_std[:, 2])
    ax10.scatter(binned_avg[:, 0], binned_avg[:, 1])
    ax10.scatter(binned_avg[:, 0], binned_avg[:, 2])
    ax10.set_xlabel('Time (h)')
    # ax10.set_yscale('log')
    ax10.set_ylabel('Fluorescent intensity (a.u.)')
    ax10.set_ylim(-10, 2000)
fig10.show()

classify_data = binned_average(sub_clf_data[0], classify_only=True)

temp_daf = pd.DataFrame(data=classify_data, columns=['time', 'green', 'red'])
temp_daf = temp_daf[temp_daf['time'] > 75]

fig11, ax11 = plt.subplots(1, 1)
sns.stripplot(data=temp_daf[temp_daf['time'] > 75], x='time', y='red', ax=ax11)
sns.violinplot(data=temp_daf, x='time', y='red', ax=ax11)
labels = list(set(temp_daf['time']))
labels.sort()
labels = ['%.2f' % lb for lb in labels]
ax11.set_xticks(np.arange(len(labels)))
ax11.set_xticklabels(labels)
fig11.show()




fig12, axes12 = plt.subplots(1, 2, figsize=(12*2, 10))
for i, ax12 in enumerate(axes12):
    sub_gr = np.vstack([cells_growth_rate[cn] for cn in sub_clf_cells_name[i]])
    binned_sub_gr_mean, binned_sub_gr_std = binned_average(sub_gr, num=50)
    binned_avg = binned_sub_gr_mean
    ax12.plot(binned_sub_gr_mean[:, 0], binned_avg[:, 1], 'g')
    # ax11.plot(binned_avg[:, 0], binned_avg[:, 2], 'r')
    ax12.errorbar(binned_avg[:, 0], binned_avg[:, 1], binned_sub_gr_std[:, 1])
    # ax12.errorbar(binned_avg[:, 0], binned_avg[:, 2], binned_std[:, 2])
    ax12.scatter(binned_sub_gr_mean[:, 0], binned_avg[:, 1])
    # ax12.scatter(binned_avg[:, 0], binned_avg[:, 2])
    ax12.set_xlabel('Time (h)')
    # ax12.set_yscale('log')
    ax12.set_ylabel('Growth rate (1/h)')
    ax12.set_ylim(-40, 150)
fig12.show()



fig13, ax13 = plt.subplots(1, 1, figsize=(12*2, 10))
for i in list(sub_clf_cells_name.keys()):
    sub_gr = np.vstack([cells_growth_rate[cn] for cn in sub_clf_cells_name[i]])
    binned_sub_gr_mean, binned_sub_gr_std = binned_average(sub_gr, num=50)
    binned_avg = binned_sub_gr_mean
    ax13.plot(binned_sub_gr_mean[:, 0], binned_avg[:, 1], color=colors_4[i])
    # ax13.plot(binned_avg[:, 0], binned_avg[:, 2], 'r')
    ax13.errorbar(binned_avg[:, 0], binned_avg[:, 1], binned_sub_gr_std[:, 1], color=colors_4[i])
    # ax13.errorbar(binned_avg[:, 0], binned_avg[:, 2], binned_std[:, 2])
    ax13.scatter(binned_sub_gr_mean[:, 0], binned_avg[:, 1], color=colors_4[i])
    # ax13.scatter(binned_avg[:, 0], binned_avg[:, 2])
    ax13.set_xlabel('Time (h)')
    # ax13.set_yscale('log')
    ax13.set_ylabel('Growth rate (1/h)')
    ax13.set_ylim(-40, 150)
fig13.show()

# %% show distribution of all cells' size
# fig1, ax = plt.subplots(1, 1, figsize=(12, 10))
# ax.hist(np.log(fd_dfs['area']), bins=100)
# ax.set_xlabel('Cell area (log(pixels))')
# ax.set_ylabel('Number')
# fig1.show()
# # %%
#
# # fig1, ax = plt.subplots(1, 2, figsize=(21*2, 10*2))
# cells = list(set(fd_dfs['chamber']))
# cells = np.random.choice(cells, 6)
# fig2, ax = plt.subplots(1, 1, figsize=(18, 10))
# for cell in cells:
#     ax.scatter(fd_dfs[fd_dfs['chamber'] == cell]['time_h'],
#                fd_dfs[fd_dfs['chamber'] == cell]['area'],
#                s=158)
# ax.set_xlim(fd_dfs['time_h'].min() + np.ptp(fd_dfs['time_h']) * 0.2,
#             fd_dfs['time_h'].min() + np.ptp(fd_dfs['time_h']) * 0.8)
# ax.set_xlabel('Time (h)')
# ax.set_ylabel('Cell size (pixels)')
# fig2.show()
#
# # %% extract data with fluorescence information
# ff_dfs = fd_dfs[~np.isnan(fd_dfs['green_mean'])]
# fig1, ax = plt.subplots(1, 2, figsize=(21, 10))
# cells = list(set(ff_dfs['chamber']))
# cells = np.random.choice(cells, 3)
# for cell in cells:
#     ax[0].plot(ff_dfs[ff_dfs['chamber'] == cell]['time_h'],
#                ff_dfs[ff_dfs['chamber'] == cell]['green_mean'] / ff_dfs[ff_dfs['chamber'] == cell]['red_mean'])
#     # ax[0].plot(ff_dfs[ff_dfs['chamber'] == cell]['time_h'], ff_dfs[ff_dfs['chamber'] == cell]['red_mean'], '-r')
# # ax[0].set_ylim(100, 50000)
# ax[0].set_yscale('log')
# ax[0].set_xlim(0)
# ax[0].set_xlabel('Time (h)')
# ax[0].set_ylabel('Fold increase (gfp/rfp)')
# ax[0].grid(False)
#
# for cell in cells:
#     ax[1].scatter(ff_dfs[ff_dfs['chamber'] == cell]['red_mean'], ff_dfs[ff_dfs['chamber'] == cell]['green_mean'],
#                   s=20)
# ax[1].set_xscale('log')
# ax[1].set_xlim(0.1, 5000)
# ax[1].set_yscale('log')
# ax[1].set_ylim(0.1, 15000)
# ax[1].set_xlabel('Rfp intensity')
# ax[1].set_ylabel('Gfp intensity')
# ax[1].grid(False)
# fig1.show()
# %%
# cells = np.array(list(set(ff_dfs['chamber'])))
# cell_ints = [True if ff_dfs[ff_dfs['chamber'] == cell]['time_h'].max() > 10 else False for cell in cells]
# fc_list = []
# for cell in tqdm(cells[cell_ints]):
#     fc_i = ff_dfs[ff_dfs['chamber'] == cell].iloc[0]['green_mean'] / ff_dfs[ff_dfs['chamber'] == cell].iloc[0][
#         'red_mean']
#     fc_f = ff_dfs[ff_dfs['chamber'] == cell].iloc[-1]['green_mean'] / ff_dfs[ff_dfs['chamber'] == cell].iloc[-1][
#         'red_mean']
#     fc = fc_i / fc_f
#     fc_list.append(fc)
#
# fc = pd.DataFrame(data=dict(chamber=cells[cell_ints],
#                             fold_change=fc_list))
# fc = fc[~np.isinf(fc['fold_change'])]
# fc = fc[fc['fold_change'] > 0]
# fig3, ax = plt.subplots(1, 1, figsize=(12, 12))
# sns.histplot(data=fc, x='fold_change', bins=200, log_scale=True, ax=ax)
#
# fig3.show()
#
# # %% BIN AVERGATE
#
# index_num = 99
# ff_dfs.index = pd.Index(range(len(ff_dfs)))
# ff_dfs['fold_change'] = ff_dfs['green_mean'] / ff_dfs['red_mean']
# cells = list(set(ff_dfs['chamber']))
# ints = [True if ff_dfs[ff_dfs['chamber'] == cell]['time_h'].max() > 40 else False for cell in cells]
# ints_2 = [True if ff_dfs[ff_dfs['chamber'] == cell]['fold_change'].values[-1] < 4 else False for cell in cells]
# cell_ints = np.logical_and(ints, ints_2)
# cell_ints = np.where(cell_ints)[0]
# cells = np.array(cells)
# traj_mask = [True if cham in cells[list(cell_ints)] else False for cham in ff_dfs['chamber']]
# traj_pd = ff_dfs[traj_mask]
# time_min = traj_pd['time_h'].min()
# time_max = traj_pd['time_h'].max()
# time_intervals = np.linspace(time_min, time_max, num=index_num + 1)
# bin_mean = []
# for i in range(index_num):
#     bin_avg = \
#         traj_pd[np.logical_and(traj_pd['time_h'] >= time_intervals[i], traj_pd['time_h'] < time_intervals[i + 1])][
#             'fold_change']
#     bin_mean.append(np.mean(bin_avg))
# time_mean = []
# for i in range(index_num):
#     time_mean.append(np.mean([time_intervals[i], time_intervals[i + 1]]))
#
# fi2, ax = plt.subplots(1, 1)
# for cell in cells:
#     ax.plot(traj_pd[traj_pd['chamber'] == cell]['time_h'], traj_pd[traj_pd['chamber'] == cell]['fold_change'],
#             alpha=0.9,
#             ls='-', c='#CCD1D1', lw=1)
# ax.set_ylim(0.1, 100)
# ax.set_yscale('linear')
# ax.set_xlim(time_min, time_max)
# ax.plot(time_mean, bin_mean)
# ax.grid(False)
# fi2.show()
#
# # %%
# final_flu = []
# for cell in cells:
#     final_flu.append([ff_dfs[ff_dfs['chamber'] == cell]['green_mean'].values[-1],
#                       ff_dfs[ff_dfs['chamber'] == cell]['red_mean'].values[-1]])
# final_flu_pd = pd.DataFrame(final_flu)
# final_flu_pd['chamber'] = cells
# final_flu_pd.columns = pd.Index(['green_mean', 'red_mean', 'chamber'])
# fig4, ax = plt.subplots(1, 1)
# sns.histplot(x=final_flu_pd['red_mean'], y=final_flu_pd['green_mean'], bins=200, log_scale=True, ax=ax)
# ax.set_ylim(100, 20000)
# # ax.set_yscale('log')
# ax.set_xlim(100, 2000)
# # ax.set_xscale('log')
# fig4.show()
# # %% gaussian mixture
# from sklearn.mixture import GaussianMixture
#
# gm = GaussianMixture(n_components=6)
# gm.fit(final_flu_pd[['green_mean', 'red_mean']])
# predict = gm.predict(final_flu_pd[['green_mean', 'red_mean']])
# final_flu_pd['classify'] = predict
# fig4, ax = plt.subplots(1, 1)
# sns.histplot(x=final_flu_pd['red_mean'], y=final_flu_pd['green_mean'], bins=200, log_scale=True, ax=ax)
# ax.scatter(final_flu_pd[final_flu_pd['classify'] == 0]['red_mean'],
#            final_flu_pd[final_flu_pd['classify'] == 0]['green_mean'],
#            s=1, c='r')
# ax.set_ylim(100, 20000)
# # ax.set_yscale('log')
# ax.set_xlim(100, 2000)
# # ax.set_xscale('log')
# fig4.show()
#
# # %% PCA
# # ff_dfs = fd_dfs[~np.isnan(fd_dfs['green_mean'])]
# # cells = list(set(ff_dfs['chamber'].values))
# # cell_ints = [True if ff_dfs[ff_dfs['chamber'] == cell]['time_h'].max() > 40 else False for cell in cells]
# # data = []
# # for cell in tqdm(cells):
# #     data.append(ff_dfs[ff_dfs['chamber'] == cell]['green_mean']/ff_dfs[ff_dfs['chamber'] == cell]['red_mean'].astype(np.float))
# # data = np.array(data)
# # from sklearn.decomposition import PCA
# # pca = PCA(n_components=2)
# # pca.fit(data)
#
#
# # %%
# ff_dfs = ff_dfs[ff_dfs['red_medium'] > 0]
# ff_dfs = ff_dfs[ff_dfs['green_medium'] > 0]
#
# fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
# sns.histplot(x=ff_dfs['red_mean'], y=ff_dfs['green_mean'],
#              log_scale=(True, True),
#              cbar=True,
#              cbar_kws=dict(shrink=.75),
#              pthresh=.1,
#              ax=ax)
# ax.set_xlabel('Rfp intensity (a.u.)')
# ax.set_ylabel('Gfp intensity (a.u.)')
#
# fig2.show()
#
# # %%
# df = pd.read_csv(r'H:\ZJW_CP\20201227\fov_41_statistic.csv')
# df = df[~np.isnan(df['green_mean'])]
# cells = list(set(df['chamber']))
# # df = df[df['chamber'] == 'ch_0']
#
# fig1, ax = plt.subplots(1, 2, figsize=(18, 10))
# for cell in cells:
#     ax[0].plot(df[df['chamber'] == cell]['time_s'],
#                df[df['chamber'] == cell]['green_mean'] / df[df['chamber'] == cell]['red_mean'], '--y')
#     # ax[0].plot(df[df['chamber'] == cell]['time_s'], df[df['chamber'] == cell]['red_mean'], '-r')
# # ax[0].set_ylim(100, 50000)
# ax[0].set_yscale('log')
#
# for cell in cells:
#     ax[1].scatter(df[df['chamber'] == cell]['red_mean'], df[df['chamber'] == cell]['green_mean'])
# ax[1].set_xscale('log')
# ax[1].set_xlim(100, 2000)
# ax[1].set_yscale('log')
#
# ax[1].set_ylim(100, 10000)
#
# fig1.show()
#
# # %%
# # df = pd.read_csv(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_55_statistic.csv')
# cells = list(set(dfs['chamber']))
# fig2, ax = plt.subplots(1, 1, figsize=(18, 10))
# for cell in cells:
#     ax.plot(dfs[dfs['channel'] == cell]['time_h'], dfs[dfs['channel'] == cell]['area'])
#
# # ax.set_yscale('log')
# ax.set_xlim(dfs['time_h'].min() + np.ptp(dfs['time_h']) * 0.0,
#             dfs['time_h'].min() + np.ptp(dfs['time_h']) * 1)
# fig2.show()
# # fig1, ax = plt.subplots(1, 1)
# # for cell in cells:
# #     ax.plot(df[df['channel'] == cell]['red_mean'], df[df['channel'] == cell]['green_mean'])
# # ax.set_xscale('log')
# # ax.set_xlim(100, 2000)
# # ax.set_yscale('log')
# # ax.set_ylim(100, 10000)
# # fig1.show()
# # %%
# db = load(r'X:\chupan\mother machine\20201225_NCM_pECJ3_M5_L3\fov_1.jl')
# fig2, ax = plt.subplots(1, 1)
# ax.imshow(db['chamber_cells_mask']['ch_1'][5])
#
# fig2.show()
