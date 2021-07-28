# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
# Libs
import pandas as pd
import numpy as np
# Own modules
import momo_pipeline as mp
from joblib import dump
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
    threading_working[thread_init] = True
    obj.process_flow_CPU()
    obj.relink_cells_musk()
    exitthread[thread_init] = True
    threading_working[thread_init] = False
    del obj
    return None


# %%
if __name__ == '__main__':
#%%
    THREADING = True
    THREADING_Limit = 2
    print('[Momo] -> Loading Files')
    # DIR = r'/media/fulab/4F02D2702FE474A3/MZX'
    DIR = r"/home/fulab/data2/ZZ"
    fovs_name = mp.get_fovs(DIR, time_step=60*4)
    fovs_num = len(fovs_name)
    exitthread = [False] * fovs_num
    threading_working = [False] * fovs_num
    to_process = None
    init = 0
    if THREADING:
        while fovs_name:
            while np.sum(threading_working) >= THREADING_Limit:
                time.sleep(5)
            to_process = fovs_name.pop(0)
            print(f'Processing {init + 1}/{fovs_num}')
            to_process.process_flow_GPU()
            thread.start_new_thread(thread_dump, (to_process, init))
            time.sleep(1)
            init += 1

        del to_process

        while False in exitthread:
            time.sleep(5)
    else:
        while fovs_name:
            to_process = fovs_name.pop(0)
            print(f'Processing {init + 1}/{fovs_num}')
            to_process.process_flow_GPU()
            to_process.process_flow_CPU()
            init += 1

        del to_process

    all_scv_name = [file for file in os.listdir(DIR) if (file.split('.')[-1] == 'csv')]
    all_scv = [dask.delayed(paral_read_csv)(ps) for ps in all_scv_name]
    with ProgressBar():
        al_df = dask.compute(*all_scv, scheduler='threads')
    cells_dict = {}
    print('dump dic of mother cells data.\n')
    for df in tqdm(al_df):
        cells_name = list(set(df['chamber']))
        for na in cells_name:
            cells_df = df[df['chamber'] == na]
            cells_df = cells_df[cells_df['area'] > 2]
            cells_df['time_h'] = [convert_time(s) for s in cells_df['time_s'] - cells_df['time_s'].min()]
            cells_dict.update({na: cells_df})

    dump(cells_dict, os.path.join(DIR, 'mothers_raw_dic.jl'), compress='lz4')


