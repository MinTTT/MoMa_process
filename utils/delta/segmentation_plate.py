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
inputs_folder = os.path.join(DeLTA_data, 'plate_image', 'phase')  # run bioformats2sequence.py first
outputs_folder = os.path.join(DeLTA_data, 'evaluation', 'plate_mask')
model_file = os.path.join(DeLTA_data, 'models', 'Unet_plate.hdf5')
input_files = os.listdir(inputs_folder)

# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)
origin_size = (2472, 3296)
# Load up model_for_colony:
model = unet_chambers(input_size=input_size)
model.load_weights(model_file)

# Predict:
predGene = predictGenerator_seg(inputs_folder, files_list=input_files, target_size=target_size)
results = model.predict_generator(predGene, len(input_files), verbose=1)

# Post process results:
results[:, :, :, 0] = postprocess(results[:, :, :, 0], min_size=1000)
reshape_size = np.zeros(shape=(results.shape[0],) + origin_size + (1,), dtype=bool)

for image_index in tqdm(range(len(results))):
    reshape_size[image_index, :, :, 0] = trans.resize(results[image_index, :, :, 0], output_shape=origin_size)
# Save to disk:
saveResult_seg(outputs_folder, reshape_size, files_list=input_files)

# Fit the plate
plate_box = []
for image_index in tqdm(range(len(results))):
    contour_plate = find_contours(reshape_size[image_index, :, :, 0])[0]
    xtl, ytl, xbr, ybr = contour_plate[:, 1].min(), contour_plate[:, 0].min(), \
                         contour_plate[:, 1].max(), contour_plate[:, 0].max()
    box = dict(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)

    plate_box.append(box)

# Crop plate region

plate_seg_folder = os.path.join(DeLTA_data, 'evaluation', 'plate_seg')

for image_index in tqdm(range(len(results))):
    x_length, y_length = int(plate_box[image_index]['xbr'] - plate_box[image_index]['xtl']), \
                         int(plate_box[image_index]['ybr'] - plate_box[image_index]['ytl'])
    x_grid = np.linspace(plate_box[image_index]['xtl'], plate_box[image_index]['xbr'], num=4, dtype=int)
    y_grid = np.linspace(plate_box[image_index]['ytl'], plate_box[image_index]['ybr'], num=4, dtype=int)
    x_grids, y_grids = np.meshgrid(x_grid, y_grid)
    xtls = x_grids[:-1, :-1]
    ytls = y_grids[:-1, :-1]
    x_grids_index, y_grids_index = np.meshgrid(range(4), range(4))
    x_indexes = x_grids_index[:-1, :-1].flatten()
    y_indexes = y_grids_index[:-1, :-1].flatten()
    image = tifflib.imread(os.path.join(inputs_folder, input_files[image_index]))
    for sub_image_index in range(9):
        sub_box = dict(xtl=x_grids[x_indexes[sub_image_index], y_indexes[sub_image_index]],
                       ytl=y_grids[x_indexes[sub_image_index], y_indexes[sub_image_index]],
                       xbr=x_grids[x_indexes[sub_image_index]+1, y_indexes[sub_image_index]+1],
                       ybr=y_grids[x_indexes[sub_image_index]+1, y_indexes[sub_image_index]+1])
        sub_image = cropbox(image, sub_box)
        tifflib.imwrite(os.path.join(plate_seg_folder,
                                     input_files[image_index].strip(input_files[image_index].split('.')[-1])+
                                     f'{sub_image_index}.tiff'), data=sub_image)



# #%%
# ps = r'F:\PHA_library\train_set\colony_mask'
#
# masks = os.listdir(ps)
#
# for file in masks:
#     image = tifflib.imread(os.path.join(ps, file))
#     image[image == 1] = 255
#
#     image.astype(np.uint8)
#     tifflib.imwrite(os.path.join(ps, file), image)