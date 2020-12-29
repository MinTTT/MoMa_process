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
# […]

# Own modules
from utils.delta.data import saveResult_seg, predictGenerator_seg, postprocess
from utils.delta.model import unet_chambers, unet_seg
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tifffile as tif
from utils.delta.utilities import getChamberBoxes, getDriftTemplate, driftcorr, rangescale, cropbox
from skimage.io import imread
from utils.delta.utilities import cropbox
from skimage import io
from tqdm import tqdm
from utils.rotation import rotate_fov

# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# %% crop files to 512 * 2048 and rotate the channels to vertical.

files_folder = r'F:\ZJW_CP\Temp_01_MOPS_GLU_PECJ35_M5_002\single_fov\Temp_01_MOPS_GLU_PECJ35_M5_002_v5/'
file_list = os.listdir(files_folder)
save_folder = r'F:/ZJW_CP/cropped_fov/'

try:
    os.makedirs(save_folder)
except FileExistsError:
    print(f'Folder {save_folder} has existed!\n')



roi = dict(xtl=0,
           xbr=2048,
           ytl=225,
           ybr=737)

for im_name in tqdm(file_list):
    im = io.imread('/'.join([files_folder, im_name]))
    im_ro, _ = rotate_fov(np.expand_dims(im, axis=0), crop=False)
    im_crp = cropbox(im_ro.squeeze(), roi)
    im_crp = im_crp[::-1, ...]
    io.imsave('/'.join([save_folder, im_name]), im_crp, check_contrast=False)

# %% This part start to predict channels
TEST_data_dir = r'F:\ZJW_CP/'
seg_folder_name = TEST_data_dir + 'cropped_fov/'
outputs_folder = TEST_data_dir + 'cropped_fov_channel_sg/'
side_channel_folder = TEST_data_dir + 'cropped_fov_side_channels_sg/'


model_file = r'test_data_set/model/chambers_id_tessiechamp_old.hdf5'
seg_model_file = r'./test_data_set/model/unet_moma_seg_multisets.hdf5'
seg_output_folder = TEST_data_dir + 'cropped_fov_cell_fg/'
unprocessed = os.listdir(seg_folder_name)
min_chamber_area = 5e2

try:
    os.makedirs(side_channel_folder)
except FileExistsError:
    pass

try:
    os.makedirs(seg_output_folder)
except FileExistsError:
    pass


target_size = (512, 512)
input_size = target_size + (1,)
process_size = 120

#%%
# create output dirs if need
try:
    os.makedirs(outputs_folder)
    print(f'Folder {outputs_folder} was created.')
except FileExistsError:
    print(f'Folder {outputs_folder} was existed.')

model = unet_chambers(input_size=input_size)
model.load_weights(model_file)

while (unprocessed):
    ps = min(process_size, len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]

    preGene = predictGenerator_seg(seg_folder_name, files_list=to_process, target_size=target_size)

    results = model.predict_generator(preGene, len(to_process), verbose=1)

    results[..., 0] = postprocess(results[..., 0], min_size=min_chamber_area)

    saveResult_seg(outputs_folder, results, files_list=to_process)

# %% load phase contract image and their channel masks

fovs_name = os.listdir(seg_folder_name)
ims = np.array([imread(seg_folder_name + name) for name in fovs_name])
channel_masks = np.array([imread(outputs_folder + name) for name in fovs_name])
ImageSize = (512, 2048)
TargetSize = (512, 512)
channel_masks = cv2.resize(np.moveaxis(channel_masks, 0, -1), ImageSize[::-1])  # rescaled masks have a yxt order
channel_masks = np.moveaxis(channel_masks, -1, 0)
# channel_masks_fil = np.copy(channel_masks)
# channel_masks_fil = postprocess(channel_masks_fil.astype(np.float32), square_size=5, min_size=8e3)
channel_boxes = getChamberBoxes(channel_masks[0, ...])



# %%  image drift correction

channel_boxes = getChamberBoxes(channel_masks[50, ...])

# crop partial image including channels as the matching template
drift_template = getDriftTemplate(channel_boxes, ims[50, ...])
driftcorbox = dict(xtl=0,
                   xbr=None,
                   ytl=0,
                   ybr=max(channel_boxes, key=lambda elem: elem['ytl'])['ytl']  # find the maximum ytl
                   )
traferd_images, driftvalues = driftcorr(ims, template=drift_template, box=driftcorbox)



#%% cell segmentation

results = ims[0:5]
fig1, ax1 = plt.subplots(len(results), 1)
for i, ax in enumerate(ax1):
    ax.imshow(results[i, ...])
fig1.show()

fig2, ax2 = plt.subplots(1, 1)
ax2.imshow(drift_template)
fig2.show()


#%%
seg_inputs = []
# Compile segmentation inputs: i * m, i: time point, m channels
for m, chamberbox in enumerate(tqdm(channel_boxes)):
    for i in range(traferd_images.shape[0]):
        side_channel = cv2.resize(rangescale(cropbox(traferd_images[i], chamberbox), (0, 1)),
                                     (32, 256))
        tif.imsave(side_channel_folder + f'Temp_01_MOPS_GLU_PECJ35_M5_002_v5_c0_chan{m}_t{i}.tif', side_channel)
        seg_inputs.append(side_channel)  # TODO: There are some things need to consider
seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)  # Format into 4D tensor





#%%
# Run segmentation U-Net:
target_size_seg = (256, 32)
model_seg = unet_seg(input_size=target_size_seg + (1,))
model_seg.load_weights(seg_model_file)

seg = model_seg.predict(seg_inputs, verbose=1)
seg = postprocess(seg[:, :, :, 0])
j = 0
for m, chamberbox in enumerate(channel_boxes):
    for i in range(traferd_images.shape[0]):
        tif.imsave(seg_output_folder + f'Temp_01_MOPS_GLU_PECJ35_M5_002_v5_c0_chan{m}_t{i}.tif', seg[j, ...])
        j += 1


#%%
results = seg[54:64, ...]
seg_results = seg_inputs[54:64, ...]
fig4, ax4 = plt.subplots(1, results.shape[0], figsize=(18, 10))
for i, ax in enumerate(ax4):
    ax.imshow(results[i, ...])
fig4.show()
fig3, ax3 = plt.subplots(1, results.shape[0], figsize=(18, 10))
for i, ax in enumerate(ax3):
    ax.imshow(seg_results[i, ...].squeeze())
fig3.show()