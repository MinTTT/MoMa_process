# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
# Libs
import pandas as pd
# Own modules
import momo_pipeline as mp
from joblib import dump
import cv2
import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm

def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


def to_BGR(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def paral_read_csv(ps):
    return pd.read_csv(os.path.join(DIR, ps))

# %%

DIR = r'G:\ubuntu_data\20210101_NCM_pECJ3_M5_L3'
fovs_name = mp.get_fovs_name(DIR)
fovs_num = len(fovs_name)
init = 0
while fovs_name:
    to_process = fovs_name.pop(0)
    print(f'Processing {init + 1}/{fovs_num}')
    init += 1
    to_process.process_flow()

all_scv_name = [file for file in os.listdir(DIR) if (file.split('.')[-1] == 'csv' and file.split('_')[0] == 'fov')]
all_scv = [dask.delayed(paral_read_csv)(ps) for ps in all_scv_name]
with ProgressBar():
    al_df = dask.compute(*all_scv, scheduler='threads')
cells_dict = {}
print('dump dic of mother cells data.\n')
for df in tqdm(al_df):
    cells_name = list(set(df['chamber']))
    for na in cells_name:
        cells_df = df[df['chamber'] == na]
        cells_df = cells_df[cells_df['area'] > 100]
        cells_df['time_h'] = [convert_time(s) for s in cells_df['time_s'] - min(cells_df['time_s'])]
        cells_dict.update({na: cells_df})

dump(cells_dict, os.path.join(DIR, 'mothers_raw_dic.jl'), compress='lz4')

# with ProgressBar():
#     al_df = dask.compute(*all_scv, scheduler='threads')
#
# dfs = pd.concat(al_df)
# dfs.index = pd.Index(range(len(dfs)))
# dfs['time_h'] = [convert_time(s) for s in dfs['time_s'] - min(dfs['time_s'])]
#
# fd_dfs = dfs[dfs['area'] > 100]
# print(f'''all chambers {len(list(set(fd_dfs['chamber'])))}''')
# fd_dfs.to_csv(os.path.join(DIR, 'all_data.csv'))
