# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
import tifffile as tiff
# […]

# Own modules
import momo_pipeline as mp
from utils.delta.utilities import cropbox
from matplotlib import pylab
import cv2
import asyncio


import dask

from dask.diagnostics import ProgressBar

def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


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
# %%
DIR = r'./test_data_set/test_data'

fovs_name = mp.get_fovs_name(DIR)

# for i, fov in enumerate(fovs_name):
#     print(f'Processing {i + 1}/{len(fovs_name)}')
#     fov.process_flow()
fovs_name = [None] + fovs_name + [None]

for i in range(len(fovs_name)-1):
    asyncio.run(asy_pip(fovs_name[i], fovs_name[i+1]))

all_scv = [file for file in os.listdir(DIR) if file.split('.')[-1] == 'csv']
dfs = [dask.delayed(paral_read_csv)(ps) for ps in all_scv]
with ProgressBar():
    dfs = dask.compute(*dfs)

dfs = pd.concat(dfs)
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
# %%



# %%

def to_BGR(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


time, ch = (0, 8), 5

ch_na = fov1.loaded_chamber_name[ch]
ch_index = fov1.index_of_loaded_chamber[ch]
if not isinstance(time, int):
    time = slice(*time)
    im = tiff.imread([os.path.join(fov1.dir, fov1.fov_name, 'phase', name) for name in fov1.times['phase'][time]])
else:
    im = tiff.imread([os.path.join(fov1.dir, fov1.fov_name, 'phase', fov1.times['phase'][time])])
    np.expand_dims(im, axis=0)

if fov1.chamber_direction == 0:
    im = im[:, ::-1, :]
channel_box = fov1.chamberboxes[ch_index]
channel_im = mp.crop_images(im, channel_box)
# channel_im = [cv2.cvtColor(channel_im[i, ...], cv2.COLOR_GRAY2BGR) for i in range(len(channel_im))]

cell_cuntour = fov1.chamber_cells_contour[ch_na][time]
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
