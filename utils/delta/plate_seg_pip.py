# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
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
"""
This script will run the plate identification/segmentation U-Net.

@author: Pan Marcus CHU
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.delta.data import saveResult_seg, predictGenerator_seg, postprocess
from utils.delta.model import unet_chambers
import os
import skimage.transform as trans
from tqdm import tqdm
from skimage.measure import find_contours
from utils.delta.utilities import cropbox, rangescale
import tifffile as tifflib
import skimage.draw as draw

# %%
# Files:
DeLTA_data = r'F:\PHA_library'

inputs_folder = r'Z:\panchu\image\Colony_Test'  # run bioformats2sequence.py first
outputs_folder = r'Z:\panchu\image\Colony_Test\output'
colony_model_file = os.path.join(DeLTA_data, 'models', 'Unet_colony.hdf5')
plate_model_file = os.path.join(DeLTA_data, 'models', 'Unet_plate.hdf5')

# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)
origin_size = (2472, 3296)
# Load up model_for_colony:
model_for_colony = unet_chambers(input_size=input_size)
model_for_colony.load_weights(colony_model_file)
model_for_plate = unet_chambers(input_size=input_size)
model_for_plate.load_weights(plate_model_file)
# %%
# Predict plate
input_plate_files = os.listdir(inputs_folder)
predGene = predictGenerator_seg(inputs_folder, files_list=input_plate_files,
                                target_size=target_size)
results_plate = model_for_plate.predict_generator(predGene, len(input_plate_files), verbose=1)
results_plate[:, :, :, 0] = postprocess(results_plate[:, :, :, 0], min_size=1000)
# Post process results:
reshape_plate_mask = np.zeros(shape=(results_plate.shape[0],) + origin_size, dtype=bool)

# Save to disk:
# saveResult_seg(outputs_folder, reshape_size, files_list=input_plate_files)

# Fit the plate
plate_box = []
for image_index in tqdm(range(len(input_plate_files))):
    reshape_img = trans.resize(results_plate[image_index, :, :, 0], output_shape=origin_size)
    reshape_plate_mask[image_index, :, :] = reshape_img
    contour_plate = find_contours(reshape_img)[0]
    xtl, ytl, xbr, ybr = contour_plate[:, 1].min(), contour_plate[:, 0].min(), \
                         contour_plate[:, 1].max(), contour_plate[:, 0].max()
    box = dict(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)

    plate_box.append(box)
colony_mask = np.zeros(shape=(len(input_plate_files),) + origin_size)
for image_index in tqdm(range(len(input_plate_files))):
    x_length, y_length = int(plate_box[image_index]['xbr'] - plate_box[image_index]['xtl']), \
                         int(plate_box[image_index]['ybr'] - plate_box[image_index]['ytl'])
    x_grid = np.linspace(plate_box[image_index]['xtl'], plate_box[image_index]['xbr'], num=4, dtype=int)
    y_grid = np.linspace(plate_box[image_index]['ytl'], plate_box[image_index]['ybr'], num=4, dtype=int)
    x_grids, y_grids = np.meshgrid(x_grid, y_grid)
    # xtls = x_grids[:-1, :-1]
    # ytls = y_grids[:-1, :-1]
    x_grids_index, y_grids_index = np.meshgrid(range(4), range(4))
    x_indexes = x_grids_index[:-1, :-1].flatten()
    y_indexes = y_grids_index[:-1, :-1].flatten()
    image = tifflib.imread(os.path.join(inputs_folder, input_plate_files[image_index]))
    image = rangescale(image, (0, 1))
    sub_images = []
    sub_images_shape = []
    sub_boxes = []
    reshaped_sub_images = np.zeros(shape=(9,) + target_size + (1,))
    for sub_image_index in range(9):
        sub_box = dict(xtl=x_grids[x_indexes[sub_image_index], y_indexes[sub_image_index]],
                       ytl=y_grids[x_indexes[sub_image_index], y_indexes[sub_image_index]],
                       xbr=x_grids[x_indexes[sub_image_index] + 1, y_indexes[sub_image_index] + 1],
                       ybr=y_grids[x_indexes[sub_image_index] + 1, y_indexes[sub_image_index] + 1])
        sub_boxes.append(sub_box)
        sub_img = cropbox(image, sub_box)
        sub_images.append(cropbox(image, sub_box))
        sub_images_shape.append(sub_img.shape)
        reshaped_sub_images[sub_image_index, :, :, 0] = trans.resize(sub_img, target_size, order=1, anti_aliasing=True)

    sub_img_masks = model_for_colony.predict(reshaped_sub_images)
    # sub_img_masks[:, :, :, 0] = postprocess(sub_img_masks[:, :, :, 0], min_size=50)

    for index, sub_box in enumerate(sub_boxes):
        colony_mask[image_index, sub_box['ytl']:sub_box['ybr'], sub_box['xtl']:sub_box['xbr']] = \
            trans.resize(sub_img_masks[index, :, :, 0], output_shape=sub_images_shape[index], order=1)

colony_mask = postprocess(colony_mask, min_size=10)


# %%
# colony_process


def measure_colony(image, contour):
    mask = draw.polygon2mask(image.shape, contour)
    size = np.sum(mask)
    strength = image[mask == True].sum()
    intensity = strength / size
    return dict(size=size, strength=strength, intensity=intensity)


colonies = []
for image_index in tqdm(range(len(input_plate_files)), desc='Measuring colonies'):
    fluor_image = tifflib.imread(os.path.join(inputs_folder, input_plate_files[image_index]))
    colonies_poly = find_contours(colony_mask[image_index, ...])
    if colonies_poly:
        for col_poly in colonies_poly:
            col_metrics = measure_colony(fluor_image, col_poly)
            col_metrics['polygon'] = col_poly
            col_metrics['plate_index'] = image_index
            colonies.append(col_metrics)
    else:
        raise Warning(f'Plate {input_plate_files[image_index]} has no colony.')

colonies_df = pd.DataFrame(data=None, columns=['size', 'strength', 'intensity', 'polygon', 'plate_index'])
colonies_df = colonies_df.append(colonies)
colonies_df_out = colonies_df[['size', 'strength', 'intensity', 'plate_index']]
# colonies_df.to_csv(os.path.join(DeLTA_data, 'evaluation', 'pick_colonies', 'statistics_colonies.csv'))
plate_name = [input_plate_files[plate_i] for plate_i in colonies_df['plate_index']]
plate_df = pd.DataFrame(data=dict(plate_name=plate_name))
colonies_df_out = pd.merge(colonies_df_out, plate_df, left_index=True, right_index=True)
colonies_df_out.to_csv(os.path.join(outputs_folder, 'statistics_colonies.csv'))
# %%

high_indexes = []
for plate_index in range(len(input_plate_files)):

    phase_image = tifflib.imread(os.path.join(inputs_folder, input_plate_files[plate_index]))
    contour_image = np.zeros(origin_size)

    colonies_intensity = colonies_df[colonies_df['plate_index'] == plate_index]['intensity']
    high_index = colonies_intensity.sort_values(ascending=False).index
    high_indexes.append(high_index.to_list())
    fig1, ax1 = plt.subplots(1, 1, figsize=(40, 30))
    for index, poly in enumerate(colonies_df.iloc[high_index]['polygon']):
        r, c = poly.T
        rr, cc = draw.polygon_perimeter(r, c, shape=origin_size)
        contour_image[rr, cc] = 1
        # ax1.text(cc[0], rr[0], f'{index}, {int(colonies_df.iloc[high_index]["intensity"].iloc[index])}')
        # ax1.text(cc[0], rr[0], f'{index}')
        ax1.scatter(cc[0], rr[0], facecolors='none', edgecolors='r')

    ax1.imshow(phase_image, cmap='gray')
    ax1.imshow(contour_image, cmap='Oranges', alpha=.2)
    ax1.set_title(f'{len(colonies_intensity)} colonies')
    fig1.show()
    fig1.savefig(
        os.path.join(outputs_folder, f'{input_plate_files[plate_index].strip(".tif")}.png'))


# colonies_df_out_high = colonies_df_out.iloc[np.array(high_indexes).flatten(), :]
# colonies_df_out_high.to_csv(os.path.join(outputs_folder, 'statistics_colonies_high.csv'))

# %%
# for image_index in tqdm(range(len(results))):
#     ori_image = tifflib.imread(os.path.join(inputs_folder, input_files[image_index]))
#     reshpaed_image = trans.resize(results[image_index, :, :, 0], output_shape=ori_image.shape, order=1)
#     reshpaed_image = reshpaed_image.astype(np.uint8) * 255
#     tifflib.imwrite(os.path.join(outputs_folder, input_files[image_index]), reshpaed_image)
#     # TODO: mask has pixel shift
