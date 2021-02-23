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
import _thread as thread
import time


def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


def paral_read_csv(ps):
    return pd.read_csv(os.path.join(DIR, ps))


def thread_dump(obj: mp.MomoFov, thread_init: int) -> None:
    obj.process_flow_CPU()
    exitthread[thread_init] = True
    return None


# %%
if __name__ == '__main__':
    DIR = r'G:\ubuntu_data\20210130_pECJ3_M5_L3_2'
    fovs_name = mp.get_fovs_name(DIR)
    fovs_num = len(fovs_name)
    exitthread = [False] * fovs_num
    init = 0
    while fovs_name:
        to_process = fovs_name.pop(0)
        print(f'Processing {init + 1}/{fovs_num}')
        to_process.process_flow_GPU()
        thread.start_new_thread(thread_dump, (to_process, init))
        init += 1

    del to_process

    while False in exitthread:
        time.sleep(5)

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
            cells_df['time_h'] = [convert_time(s) for s in cells_df['time_s'] - cells_df['time_s'].min()]
            cells_dict.update({na: cells_df})

    dump(cells_dict, os.path.join(DIR, 'mothers_raw_dic.jl'), compress='lz4')


