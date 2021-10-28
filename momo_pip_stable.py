# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

import threading as thread
# Built-in/Generic Imports
import os
import time
import sys
import getopt
import dask
import numpy as np
# Libs
import pandas as pd
from dask.diagnostics import ProgressBar
from joblib import dump
from tqdm import tqdm

# Own modules
import momo_pipeline as mp


def convert_time(time_s: float):
    h, s = divmod(time_s, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h


def paral_read_csv(ps):
    return pd.read_csv(os.path.join(DIR, ps))


def thread_dump(obj: mp.MomoFov, thread_init: int) -> None:
    threading_working[thread_init] = True
    obj.process_flow_CPU()
    exitthread[thread_init] = True
    threading_working[thread_init] = False
    del obj
    return None


def opt_parse(argv):
    file_name = argv[0]
    argv = argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'i:')
    except getopt.GetoptError:
        print(f'{file_name} -i <directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ['-i']:
            return arg
    if len(args) == 1:
        return args[0]
    else:
        raise getopt.GetoptError(f'too much args. {file_name} -i <directory>')


# %%
if __name__ == '__main__':
    # %%
    THREADING = True
    THREADING_Limit = 2
    print('[Momo] -> Loading Files')
    # DIR = r'/media/fulab/4F02D2702FE474A3/MZX'
    if len(sys.argv) > 1:
        DIR = opt_parse(sys.argv)
    else:
        DIR = r"./test_data_set/test_data"
    fovs_name = mp.get_fovs(DIR, time_step=120)
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
            thread_of_dump = thread.Thread(target=thread_dump, args=(to_process, init))
            thread_of_dump.start()
            # thread.start_new_thread(thread_dump, (to_process, init))
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
