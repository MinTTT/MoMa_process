# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

import asyncio
# Built-in/Generic Imports
import os

import cv2
import dask
import numpy as np  # Or any other
# Libs
import pandas as pd
from dask.diagnostics import ProgressBar
from matplotlib import pylab

# Own modules
import momo_pipeline as mp


# […]
# […]

def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h

def to_BGR(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


async def asy_process(fov):
    if fov:
        print(f'Now, {fov.fov_name}: detect channels.\n')
        fov.detect_channels()
        print(f'Now, {fov.fov_name}: detect frameshift.\n')
        fov.detect_frameshift()
        print(f'Now, {fov.fov_name}: detect cells.\n')
        fov.cell_detection()
        print(f"Now, {fov.fov_name}: extract cells' features.\n")
        fov.extract_mother_cells_features()
        print(f"Now, {fov.fov_name}: get mother cells data.\n")
        fov.parse_mother_cell_data()
    return None


async def asy_dump(fov):
    if fov:
        print(f"Now, {fov.fov_name}: dump memory data.\n")
        fov.dump_data()
        print(f"{fov.fov_name} finished dump data.\n")
        del fov

async def asy_pip(fov1, fov2):
    tsk1 = asyncio.create_task(asy_dump(fov1))
    tsk2 = asyncio.create_task(asy_process(fov2))
    await tsk1
    await tsk2


def paral_read_csv(ps):
    return pd.read_csv(os.path.join(DIR, ps))



#%%
DIR = r'test_data_set/test_data'

fovs_name = mp.get_fovs_name(DIR)
fovs_num = len(fovs_name)
init = 0
while fovs_name:
    to_process = fovs_name.pop(0)
    print(f'Processing {init + 1}/{fovs_num}')
    init += 1
    to_process.process_flow()


all_scv = [file for file in os.listdir(DIR) if file.split('.')[-1] == 'csv']
all_scv = [dask.delayed(paral_read_csv)(ps) for ps in all_scv]

with ProgressBar():
    al_df = dask.compute(*all_scv, scheduler='threads')

dfs = pd.concat(al_df)
dfs.index = pd.Index(range(len(dfs)))
dfs['time_h'] = [convert_time(s) for s in dfs['time_s'] - min(dfs['time_s'])]

fd_dfs = dfs[dfs['area'] > 100]
print(f'''all chambers {len(list(set(fd_dfs['chamber'])))}''')
fd_dfs.to_csv(os.path.join(DIR, 'all_data.csv'))
# %%
fov1 = fovs_name[0]
fov1.detect_channels()
fov1.detect_frameshift()
fov1.cell_detection()
fov1.extract_mother_cells_features()
fov1.parse_mother_cell_data()
# %% dump jl
from joblib import load

def find_jl(dir):
    fn = [f for f in os.listdir(dir) if f.split('.')[-1] == 'jl']
    fn = [os.path.join(dir, f) for f in fn]
    return fn


DIR = r'./test_data_set/test_data'

jl_file = find_jl(DIR)

fov_jl = load(jl_file[0])


# %%



def draw_contour(ch=None, ch_name=None,channel='phase',time=0, fov_jl=None):
    if ch:
        ch_na = fov_jl['chamber_loaded_name'][ch]
        # ch_index = list(range(len(fov_jl['chamber_loaded_name'])))[ch]
    else:
        ch_na = ch_name

    channl_key = dict(phase='chamber_phase_images',
                      green='chamber_green_images',
                      red='chamber_red_images')
    channl_color = channl_key[channel]


    if not isinstance(time, int):
        time = slice(*time)
        channel_im = fov_jl[channl_color][ch_na][time]
        # im = tiff.imread([os.path.join(fov_jl.dir, fov_jl.fov_name, 'phase', name) for name in fov_jl.times['phase'][time]])
    else:
        # im = tiff.imread([os.path.join(fov_jl.dir, fov_jl.fov_name, 'phase', fov_jl.times['phase'][time])])
        channel_im = fov_jl[channl_color][ch_na][time]
        channel_im = np.expand_dims(channel_im, axis=0)


    if channel == 'phase':
        cell_cuntour = fov_jl['chamber_cells_contour'][ch_na][time]
    else:
        time_str = fov_jl['times'][channel][time]
        if isinstance(time_str, str):
            time_index = fov_jl['times']['phase'].index(time_str)
            cell_cuntour = [fov_jl['chamber_cells_contour'][ch_na][time_index]]

        else:
            time_index = [fov_jl['times']['phase'].index(ele) for ele in time_str]
            cell_cuntour = []
            for inx in time_index:
                cell_cuntour.append(fov_jl['chamber_cells_contour'][ch_na][inx])

    # channel_im_bgr = cv2.cvtColor(channel_im, cv2.CV_)

    ims_with_cnt = []
    for i, cts in enumerate(cell_cuntour):
        ims_with_cnt.append(
            cv2.drawContours(to_BGR(mp.rangescale(channel_im[i], (0, 255)).astype(np.uint8)), cts, -1,
                             (247, 220, 111),
                             1))
    ims_with_cnt = np.concatenate(ims_with_cnt, axis=1)


    return ims_with_cnt


ims_with_cnt = draw_contour(ch=4, channel='green', time=[0, 4], fov_jl=fov_jl)
pylab.imshow(ims_with_cnt)
pylab.show()

#%%

time, ch = [0, 10], 5

ch_na = fov_jl['chamber_loaded_name'][ch]
ch_index = list(range(len(fov_jl['chamber_loaded_name'])))[ch]
if not isinstance(time, int):
    time = slice(*time)
    channel_im = fov_jl['chamber_phase_images'][ch_na][time]
    # im = tiff.imread([os.path.join(fov_jl.dir, fov_jl.fov_name, 'phase', name) for name in fov_jl.times['phase'][time]])
else:
    # im = tiff.imread([os.path.join(fov_jl.dir, fov_jl.fov_name, 'phase', fov_jl.times['phase'][time])])
    channel_im = fov_jl['chamber_phase_images'][ch_na][time]
    channel_im = np.expand_dims(channel_im, axis=0)

# if fov_jl.chamber_direction == 0:
#     im = im[:, ::-1, :]
# channel_box = fov_jl.chamberboxes[ch_index]
# channel_im = mp.crop_images(im, channel_box)
# # channel_im = [cv2.cvtColor(channel_im[i, ...], cv2.COLOR_GRAY2BGR) for i in range(len(channel_im))]

cell_cuntour = fov_jl['chamber_cells_contour'][ch_na][time]
# channel_im_bgr = cv2.cvtColor(channel_im, cv2.CV_)
if not isinstance(time, int):
    ims_with_cnt = []
    for i, cts in enumerate(cell_cuntour):
        ims_with_cnt.append(
            cv2.drawContours(to_BGR(mp.rangescale(channel_im[i], (0, 255)).astype(np.uint8)), cts, -1, (247, 220, 111),
                             1))
    ims_with_cnt = np.concatenate(ims_with_cnt, axis=1)
else:
    ims_with_cnt = cv2.drawContours(mp.rangescale(channel_im, (0, 255)).astype(np.uint8), cell_cuntour, -1, 255, 1)

pylab.imshow(ims_with_cnt)
pylab.show()

# %%



