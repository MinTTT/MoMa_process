# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import sys
from joblib import delayed, Parallel, load
import seaborn as sns

from scipy.fftpack import fft, fftfreq, ifft, fftshift

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


def fft_filter(singal, window_width):
    S = fft(singal)
    fre = fftfreq(len(singal))
    H = abs(fre) < window_width
    S_filterd = S * H
    return ifft(S_filterd)


def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


def log_fitting(length, time):
    """
    log fitting cell length (area) and time, return cells' instantaneous growth rate and length growth rate
    :param length: ndarray
    :param time: ndarray
    :return: list
    """
    log_length = np.log(length).reshape(-1, 1)
    time = time.reshape(-1, 1)
    A = np.hstack([time, np.ones(log_length.shape)])
    gr_length, _ = np.linalg.lstsq(A, log_length, rcond=None)[0]
    time_length = (time.max() - time.min()) / 2 + time.min()
    time_diff = np.diff(time, axis=0).ravel()
    inst_rate = np.diff(log_length, axis=0).ravel() / time_diff
    inst_time = time[:-1, 0].ravel() + time_diff / 2
    return [gr_length, time_length, inst_rate, inst_time]


def cell_div_filter(cell_area_t, threshold=10):
    cell_area = cell_area_t[:, 0].ravel()
    time_h = cell_area_t[:, 1].ravel()
    area_dd = np.diff(np.diff(cell_area))
    outlier = np.where(np.logical_or((0.8 * cell_area[1:-1] < -area_dd),
                                     np.logical_and(0.2 * cell_area[1:-1] < -area_dd,
                                                    -area_dd < 0.3 * cell_area[1:-1])))[0] + 1
    # print(outlier)
    mask = np.array([True] * len(cell_area))
    mask[outlier] = False
    fild_cell_area = cell_area[mask]
    fild_time_h = time_h[mask]
    if threshold is not None:
        size_mask = fild_cell_area > threshold
        fild_cell_area = fild_cell_area[size_mask]
        fild_time_h = fild_time_h[size_mask]
    area_dd = np.diff(np.diff(fild_cell_area))
    outlier2 = np.where(1.2 * fild_cell_area[1:-1] < area_dd)[0] + 1
    mask = np.array([True] * len(fild_cell_area))
    mask[outlier2] = False
    fild_cell_area = fild_cell_area[mask]  # filtered data, excluding outlier
    fild_time_h = fild_time_h[mask]
    fild_area_diff = np.diff(fild_cell_area)
    div_thre = 0.7
    min_p = fild_cell_area * (1 - div_thre)
    max_p = fild_cell_area * div_thre
    filter_peaks = np.where(np.logical_and(-fild_area_diff > min_p[:-1], -fild_area_diff < max_p[:-1]))[
        0]  # division peaks
    peak_num = len(filter_peaks)
    peaks_slice = [0] + list(filter_peaks + 1) + [len(fild_cell_area)]
    num_peaks = np.diff(peaks_slice)
    # print(peaks_slice)
    len_index = np.where(num_peaks >= 4)[0]  # one cell cycle must have more than 5 records.
    div_slice_all = [slice(peaks_slice[i], peaks_slice[i + 1]) for i in range(peak_num + 1)]
    div_slice = [div_slice_all[i] for i in len_index]  # cell cycle slice
    return div_slice, fild_cell_area, fild_time_h


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

    cell_df = cell_df[[st, 'time_h']]

    cell_data = cell_df[cell_df[st] > 10].values  # if compute ref cell length, cell length < 15 are except.
    if len(cell_data[:, 1]) == 0:
        return None

    if cell_data[:, 1].max() < 60:  # time must great than 75 hrs
        return None

    div_slice, fild_cells_area, fild_time = cell_div_filter(cell_data)

    if len(div_slice) < 8:  # at least divide 9 times.
        return None
    time_all = []
    rate_all = []
    length_rate_all = []
    length_time_all = []
    for sl in div_slice:
        length = fild_cells_area[sl]
        time = fild_time[sl]
        # instantaneous growth rate
        len_rate, len_time, inst_rate, inst_time = log_fitting(length, time)

        rate_all.append(inst_rate)
        time_all.append(inst_time)
        length_rate_all.append(len_rate)
        length_time_all.append(len_time)

    rate_all = np.concatenate(rate_all)
    time_all = np.concatenate(time_all)

    length_rate_all = np.array(length_rate_all).ravel()
    length_time_all = np.array(length_time_all).ravel()

    return [np.array([time_all, rate_all]).T, np.array([length_time_all, length_rate_all]).T]


def binned_average(mat, num=100, std=True, classify_only=False):
    """
    get binned average and standard

    :param mat:
    :param num:
    :return:
    """
    binned_num = num
    binned_time = mat[:, 0]
    bins = np.linspace(binned_time.min(), binned_time.max(), num=binned_num, endpoint=False)
    index = np.searchsorted(bins, binned_time, side='right')
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


def draw_binned_growth_rate(data_dict: dict, bin_num, ax=None, **kwargs):
    """
    I'm too lazy to annotate.
    :param data_dict: dict
    :param bin_num: int
    :param kwargs:
    :return: axes
    """
    if ax is None:
        ax = plt.gca()
    cells_name = list(data_dict.keys())
    gr_array = [data_dict[na] for na in cells_name]
    gr_array = np.vstack(gr_array)
    # binned average
    bin_num = bin_num
    time = gr_array[:, 0]
    _avg, _std = binned_average(gr_array, bin_num)
    time_avg, gr_avg = _avg[:, 0], _avg[:, 1]
    time_std, gr_std = _std[:, 0], _std[:, 1]
    std_up = gr_avg + gr_std
    std_down = gr_avg - gr_std
    if 'bac_cells_num' in kwargs:
        bac_cells_num = kwargs['bac_cells_num']
    else:
        bac_cells_num = 20
    cells = np.random.choice(cells_name, bac_cells_num)
    for na in tqdm(cells):
        data = data_dict[na]
        if isinstance(data, np.ndarray):
            ax.plot(data[:, 0], data[:, 1], color='#ABB2B9', lw=0.5, alpha=0.4)
    ax.plot(data[:, 0], data[:, 1], color='#E67E22', lw=3, alpha=0.5)
    ax.scatter(time_avg, gr_avg, s=40, color='#3498DB')
    ax.plot(time_avg, std_up, '--', lw=3, color='#5DADE2')
    ax.plot(time_avg, std_down, '--', lw=3, color='#5DADE2')
    ax.set_xlim(0, time.max())
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Growth rate (1/h)')


# %% load statistic data
# ps = r'E:\Moma_statistic\20210101_NCM_pECJ3_M5_L3_mothers_raw_dic.jl'
ps = r'D:\python_code\MoMa_process\test_data_set\jl_data\mothers_raw_dic.jl'

# cells_df = cells_df.compute()
cells_df = load(ps)
cells_name = list(cells_df.keys())

# %%
cell_df = cells_df[cells_name[20]]
length = cell_df['length'] * 0.066
time_h = cell_df['time_h']

slices, fild_leng, fild_t = cell_div_filter(cell_df[['length', 'time_h']].values)

fig, ax = plt.subplots(1, 1)
ax.scatter(time_h, length, marker='d', c='#FFA2A8')
ax.plot(time_h, length, c='#808B96')
for sl in slices:
    ax.scatter(fild_t[sl], fild_leng[sl]*0.066)
ax.set_yscale('log')
ax.set_ylim(0.9, 8)
ax.set_xlim(0, 20)
ax.grid(False)
fig.show()

# %% filter data
print('calculate instantaneous growth.')
cells_growth_rate_list = Parallel(n_jobs=64, backend='threading')(
    delayed(get_growth_rate)(cells_df[cn], sizetype='length')
    for cn in tqdm(cells_name))
cells_growth_rate = {cn: cells_growth_rate_list[i] for i, cn in enumerate(cells_name)
                     if isinstance(cells_growth_rate_list[i], list)}
cells_name = list(cells_growth_rate.keys())
cells_inst_growth_rate = {cn: cells_growth_rate[cn][0] for cn in cells_name}
cells_len_growth_rate = {cn: cells_growth_rate[cn][1] for cn in cells_name}

# time range filter

mask_time = [binned_average(cells_inst_growth_rate[cn][:, 0].reshape(-1, 1), num=4) for cn in cells_name]

mask_time = [False if np.isnan(gr).any() else True for gr in mask_time]

cells_name = [cells_name[i] for i in np.arange(len(cells_name))[mask_time]]

fig1, ax = plt.subplots(1, 1)
draw_binned_growth_rate(cells_inst_growth_rate, bin_num=50, bac_cells_num=40, ax=ax)
fig1.show()
# %% binned 4
print("classify growth type.")
four_binned = Parallel(n_jobs=10, backend='threading')(
    delayed(list)(binned_average(cells_inst_growth_rate[cn], num=4, std=False)[:, 1])
    for cn in tqdm(cells_name))
# four_binned = [list(binned_average(cells_inst_growth_rate[cn][:, 0], cells_inst_growth_rate[cn][:, 1], num=4)[1])
#                for cn in cells_name]
four_binned = np.array(four_binned)

# K-MES classify
from sklearn.cluster import KMeans

km_model = KMeans(n_clusters=3, random_state=4).fit(four_binned)
km_label = km_model.labels_
colors_4 = ['#DBB38F', '#91DB8F', '#8FB7DB', '#D98FDB'] # brown, green, blue, violet
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
fig3, axes = plt.subplots(3, 1, figsize=(15, 25))
for i, ax in enumerate(axes):
    clfd_growth_dict = {cn: cells_inst_growth_rate[cn] for cn in clfd_cells[i]}
    draw_binned_growth_rate(clfd_growth_dict, bin_num=250, bac_cells_num=15, ax=ax)
    ax.set_yticks(np.arange(-0.2, 2.2, 0.2))
fig3.show()

fig3, axes = plt.subplots(3, 1, figsize=(15, 25))
for i, ax in enumerate(axes):
    clfd_growth_dict = {cn: cells_len_growth_rate[cn] for cn in clfd_cells[i]}
    draw_binned_growth_rate(clfd_growth_dict, bin_num=200, bac_cells_num=30, ax=ax)
    ax.set_yticks(np.arange(-0.1, 2.2, 0.2))
    ax.set_ylim(-0.25, 2.2)
    ax.grid(False)
fig3.show()


# %%
fluor_binnum = 50
clfd_cell = clfd_cells[1]
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
sld_cells = np.random.choice(clfd_cell, 1)
for cell in tqdm(sld_cells):
    cell_data = cells_df[cell]

    mask_df = cells_df[cell][~np.isnan(cells_df[cell]['green_medium'])]
    ax.plot(mask_df['red_medium'], mask_df['green_medium'], color='#ABB2B9', lw=0.5, alpha=0.5)
# ax.plot(mask_df['red_medium'], mask_df['green_medium'], color='#E67E22', lw=2, alpha=0.6)
ax.plot(bred_mean, bgreen_mean, lw=4, color='#3498DB')
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
    green_over_red = (flu_cells_dict[cn]['green_medium'] + 10) / (flu_cells_dict[cn]['red_medium'] + 10)
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
km_model2 = KMeans(2, random_state=3).fit(sub_log_bin_flu[:, -4:-2])
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

fig10, axes9 = plt.subplots(1, 2, figsize=(12 * 2, 10))
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
export_data = sub_clf_bin_mean_data[0]
export_flu = pd.DataFrame(data=dict(Time=binned_avg[:, 0],
                                    Green=binned_avg[:, 1],
                                    Red=binned_avg[:, 2],
                                    Green_std=binned_std[:, 1],
                                    Red_std=binned_std[:, 2]))
export_flu.to_csv(r'./exported_data/single_cell_mean_green_red.csv')


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

fig12, axes12 = plt.subplots(1, 2, figsize=(12 * 2, 10))
for i, ax12 in enumerate(axes12):
    sub_gr = np.vstack([cells_inst_growth_rate[cn] for cn in sub_clf_cells_name[i]])
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
    ax12.set_ylim(-1.5, 2)
fig12.show()

fig13, ax13 = plt.subplots(1, 1, figsize=(12 * 2, 10))
for i in list(sub_clf_cells_name.keys()):
    sub_gr = np.vstack([cells_inst_growth_rate[cn] for cn in sub_clf_cells_name[i]])
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
    ax13.set_ylim(-1, 2.5)
fig13.show()

# fov hist
fig14, ax14 = plt.subplots(2, 1, figsize=(10, 20))
for i in list(sub_clf_cells_name.keys()):
    sub_gr = sub_clf_cells_name[i]
    print(len(sub_gr))
    sub_fovs = [int(na.split('_')[1]) for na in sub_gr]
    ax14[i].hist(sub_fovs, bins=120, density=True, color=colors_4[i], alpha=0.6)
    ax14[i].set_xlabel('#FOV')
    ax14[i].set_ylabel('Ratio')
fig14.show()

fig15, axes15 = plt.subplots(1, 2, figsize=(12 * 2, 10))
for i, ax15 in enumerate(axes15):
    sub_gr = np.vstack([cells_df[cn][['time_h', 'area']].values.reshape(-1, 2) for cn in sub_clf_cells_name[i]])
    binned_sub_gr_mean, binned_sub_gr_std = binned_average(sub_gr, num=50)
    binned_avg = binned_sub_gr_mean
    ax15.plot(binned_sub_gr_mean[:, 0], binned_avg[:, 1], color=colors_4[i])
    ax15.errorbar(binned_avg[:, 0], binned_avg[:, 1], binned_sub_gr_std[:, 1], color=colors_4[i])
    ax15.scatter(binned_sub_gr_mean[:, 0], binned_avg[:, 1], color=colors_4[i])
    ax15.set_xlabel('Time (h)')
    ax15.set_ylabel('Cell area')
fig15.show()
