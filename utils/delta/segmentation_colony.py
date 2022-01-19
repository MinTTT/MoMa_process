"""
This script will run the plate identification/segmentation U-Net.

@author: Pan Marcus CHU
"""
import numpy as np

from utils.delta.data import saveResult_seg, predictGenerator_seg, postprocess
from utils.delta.model import unet_chambers
import os
import skimage.transform as trans
from tqdm import tqdm
from skimage.measure import find_contours
from utils.delta.utilities import cropbox
import tifffile as tifflib

# Files:
DeLTA_data = r'F:\PHA_library'
inputs_folder = os.path.join(DeLTA_data, 'evaluation', 'plate_seg')  # run bioformats2sequence.py first
outputs_folder = os.path.join(DeLTA_data, 'evaluation', 'colony_mask')
model_file = os.path.join(DeLTA_data, 'models', 'Unet_colony.hdf5')
input_files = os.listdir(inputs_folder)

# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)

# Load up model_for_colony:
model = unet_chambers(input_size=input_size)
model.load_weights(model_file)

# Predict:
predGene = predictGenerator_seg(inputs_folder, files_list=input_files, target_size=target_size)
results = model.predict_generator(predGene, len(input_files), verbose=1)

# Post process results:
results[:, :, :, 0] = postprocess(results[:, :, :, 0], min_size=10)


for image_index in tqdm(range(len(results))):
    ori_image = tifflib.imread(os.path.join(inputs_folder, input_files[image_index]))
    reshpaed_image = trans.resize(results[image_index, :, :, 0], output_shape=ori_image.shape)
    reshpaed_image = reshpaed_image.astype(np.uint8) * 255
    tifflib.imwrite(os.path.join(outputs_folder, input_files[image_index]), reshpaed_image)
    # TODO: mask has pixel shift





