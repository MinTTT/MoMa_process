# -*- coding: utf-8 -*-

"""
These code is modified from Delta
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
from os import listdir, makedirs
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
# […]

# Own modules

from utils.delta.data import saveResult_seg, predictGenerator_seg, postprocess
from utils.delta.model import unet_seg

#%%
TEST_data_dir = r'./test_data_set/'
seg_folder_name = TEST_data_dir + 'segmentation_set/'
outputs_folder = TEST_data_dir + 'segmentataion_results/'
model_file = TEST_data_dir + '/model_for_colony/unet_moma_seg_multisets.hdf5'
unprocessed = listdir(seg_folder_name)

#%%
# Parameters:
target_size = (256, 32)
input_size = target_size + (1,)
process_size = 4096

# Load up model_for_colony:
model = unet_seg(input_size=input_size)
model.load_weights(model_file)

# create output dirs if need
try:
    makedirs(outputs_folder)
    print(f'Folder {outputs_folder} was created.')
except FileExistsError:
    print(f'Folder {outputs_folder} was existed.')


# Process
while (unprocessed):
    # Pop out filenames
    ps = min(process_size, len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]

    # Predict:
    predGene = predictGenerator_seg(seg_folder_name, files_list=to_process, target_size=target_size)
    results = model.predict_generator(predGene, len(to_process), verbose=1)

    # Post process results:
    results[:, :, :, 0] = postprocess(results[:, :, :, 0])

    # Save to disk:
    saveResult_seg(outputs_folder, results, files_list=to_process)