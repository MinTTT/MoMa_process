# -*- coding: utf-8 -*-

"""
 This file used to extract cells' profile in nd2 file.
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
import sys

from nd2file import ND2MultiDim
import pandas as pd
import numpy as np
from utils.delta.model import unet_seg
from utils.delta.data import saveResult_seg, predictGenerator_seg, postprocess
import matplotlib.pyplot as plt
import tensorflow as tf
import tifffile as tiff
from utils.delta.utilities import getChamberBoxes, getDriftTemplate, driftcorr, rangescale, cropbox, getSinglecells
import cv2
from tqdm import tqdm
import seaborn as sns
from joblib import dump, load

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class Cell:
    def __init__(self, contour=None, fov=None, cell_paras=dict()):
        self.cell_paras = cell_paras
        self.fov = fov
        self.contour = contour


def cell_segmentation(imdir=None,
                      model_seg=None,
                      target_size_seg=(256, 256),
                      split_factor=8, **kwargs):
    """
    This function used to predict cell pixels in a given image.
    :param imdir: str. image dir, input a 2048X2048 image.
    :param model_seg: predict model.
    :return: nd array, a binary image.
    """
    if 'im' in kwargs:
        im = kwargs['im']
    else:
        im = tiff.imread(imdir).squeeze()
    im = rangescale(im, (0, 1.))
    # ==========split image into 8X8 sub-images.=================
    ims_row = np.split(im, split_factor, axis=0)
    ims = np.vstack([np.array(np.split(row, split_factor, axis=1)) for row in ims_row])
    sub_im_num = ims.shape[0]
    index = np.arange(0, sub_im_num, split_factor).astype(np.int)
    seg_list = []
    for i in index:
        seg_inputs = np.expand_dims(ims[i:(i + split_factor), ...], axis=3)
        seg = model_seg.predict(seg_inputs)
        seg = seg.squeeze()  # remove axis3
        seg_list.append(seg)

    ims = np.vstack([np.hstack(list(sub)) for sub in seg_list])
    if 'square_size' in kwargs:
        seg = postprocess(ims, square_size=kwargs['square_size'])
    elif 'min_size' in kwargs:
        seg = postprocess(ims, min_size=kwargs['min_size'])
    else:
        seg = postprocess(ims, square_size=12, min_size=300)
    return seg


def back_corrt(im: np.ndarray, bac: float) -> np.ndarray:
    im -= bac
    im[im < 0] = 0.
    return im


def parse_nd2(ND2_FILE_PS, model_seg, **kwargs):
    """

    :param ND2_FILE_PS: string, path of ND2 file, this ND2 files have fovs, three channels including phase contrast, fluor channels X2
    :param model_seg: tensorflow model
    :param kwargs: kwargs including 'split_factor'. default is 8.
    :return: a list containing Cell objs predicted from nd files.
    """
    nd_file = ND2MultiDim(ND2_FILE_PS)
    TIME_NUM = nd_file.timepointcount
    # =================== seg part ===============
    segs = []
    for im_index in tqdm(range(TIME_NUM)):
        if 'split_factor' in kwargs:
            segs.append(cell_segmentation(im=nd_file.image_singlechannel(0, im_index, 0), model_seg=model_seg,
                                          split_factor=kwargs['split_factor']))
        else:
            segs.append(cell_segmentation(im=nd_file.image_singlechannel(0, im_index, 0), model_seg=model_seg,
                                          split_factor=8))
    # =================== seg part ===============
    cells = []
    fovs_nm = len(segs)
    for seg_index in tqdm(range(fovs_nm)):
        contours = cv2.findContours(segs[seg_index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            0]  # find
        contours.sort(key=lambda elem: np.sum(np.max(elem[:, 0, 0] + np.max(elem[:, 0, 1]))))
        im_c2 = nd_file.image_singlechannel(0, seg_index, 1).astype(np.float)
        im_c2_back_mean = np.mean(im_c2[np.logical_not(segs[seg_index])])  # c2 background
        im_c2 = back_corrt(im_c2, float(im_c2_back_mean))
        im_c3 = nd_file.image_singlechannel(0, seg_index, 2).astype(np.float)
        im_c3_back_mean = np.mean(im_c3[np.logical_not(segs[seg_index])])  # c3 background
        im_c3 = back_corrt(im_c3, float(im_c3_back_mean))
        for c, contour in enumerate(contours):  # Run through cells in frame
            # Length, width, area:
            cell_mask = np.zeros(segs[seg_index].shape, np.uint8)
            cell_mask = cv2.drawContours(cell_mask, [contour], 0, 255, -1)
            rotrect = cv2.minAreaRect(contour)
            cell_pars = dict(
                length=max(rotrect[1]),
                width=min(rotrect[1]),
                area=cv2.contourArea(contour),
                c2_mean=cv2.mean(im_c2, cell_mask)[0],
                c3_mean=cv2.mean(im_c3, cell_mask)[0]
            )

            cell = Cell(contour=contour, cell_paras=cell_pars, fov=seg_index)
            cells.append(cell)
    return cells


def get_all_pars(list, key):
    """
    get cell's parameters in Cell obj.
    :param list: a list comprising Cell objs
    :param key: cell parameters
    :return: a list of cells' parameter
    """
    return [obj.cell_paras[key] for obj in list]


def get_describe(sample: list, column_keys: list) -> pd.DataFrame:
    """
    get statistics of samples.
    :param sample: a samples' list , containing a series of list comprising Cell objects.
    :param column_keys: a list comprising keys of cell parameter
    :return: a dataframe describe the Mean, STD adn Medium of samples.
    """
    dfs = []
    for cells in sample:
        data = dict()
        for key in column_keys:
            data[key] = get_all_pars(cells, key)
        df = pd.DataFrame(data=data)
        dfs.append(df)
    column_mean = [key + '_mean' for key in column_keys]
    column_std = [key + '_std' for key in column_keys]
    column_medium = [key + '_medium' for key in column_keys]
    columns = column_mean + column_std + column_medium
    data_all = []
    for df in dfs:
        data = list(df.describe().loc['mean']) + list(df.describe().loc['std']) + list(df.describe().loc['50%'])
        data_all.append(data)
    df_summary = pd.DataFrame(data=data_all,
                              columns=columns)
    return df_summary

# […]

# %% ND2 reader
ND2_FILE_DIR = r'X:/chupan/AGAR_PAD/20201214_AGARPAD/'
MODEL_FILE = r'./test_data_set/model/delta_pads_seg.hdf5'
FILE_NAME = os.listdir(ND2_FILE_DIR)
ND2_FILE_NAME = [file for file in FILE_NAME if file.split('.')[-1] == 'nd2']
LJ_FILE_NAME = [file.split('.')[0] for file in FILE_NAME if file.split('.')[-1] == 'jl']
ND2_FILE_NAME = [file for file in ND2_FILE_NAME if file.split('.')[0] not in LJ_FILE_NAME]
target_size_seg = (256, 256)
model_seg = unet_seg(input_size=target_size_seg + (1,))
model_seg.load_weights(MODEL_FILE)
sample = []
for i, file in enumerate(ND2_FILE_NAME):
    print(f'processing {i}: {file}\n')
    cells_paras = parse_nd2(ND2_FILE_DIR + file, model_seg)
    data_save = dict(file_name=file, cells_paras=cells_paras)
    dump(cells_paras, ND2_FILE_DIR + file.split('.')[0] + '.jl')
    sample.append(cells_paras)


# %% load jl file for statistic
FILE_NAME = os.listdir(ND2_FILE_DIR)
LJ_FILE_NAME = [file.split('.')[0] for file in FILE_NAME if file.split('.')[-1] == 'jl']
sample_jl = [load(ND2_FILE_DIR+name+'.jl') for name in LJ_FILE_NAME]

column_keys = ['length', 'width', 'area', 'c2_mean', 'c3_mean']
df_summary = get_describe(sample_jl, column_keys)

df_summary['sample'] = [file_name.split('.')[0] for file_name in LJ_FILE_NAME]
df_summary.to_csv(ND2_FILE_DIR + 'statistics.csv')
# ND2_FILE_PS = r'X:/chupan/AGAR_PAD/20201214_AGARPAD/220201214_GLU_GLUTAMTE_NH2_PECJ3_002.nd2'
# MODEL_FILE = r'./test_data_set/model/delta_pads_seg.hdf5'
#
# cells = parse_nd2(ND2_FILE_PS, model_seg)
# %%
# Own modules
# MODEL_FILE = r'./test_data_set/model/delta_pads_seg.hdf5'
# IMAGE_DIR = r'F:/Agar_seg/'
#
# ims_name = os.listdir(IMAGE_DIR)
# ims_c1 = [name for name in ims_name if name.split('.')[0][-2:] == 'c1']
# ims_c2 = [name for name in ims_name if name.split('.')[0][-2:] == 'c2']
# ims_c3 = [name for name in ims_name if name.split('.')[0][-2:] == 'c3']
# ims_c1_dir = [IMAGE_DIR + name for name in ims_c1]
# ims_c2_dir = [IMAGE_DIR + name for name in ims_c2]
# ims_c3_dir = [IMAGE_DIR + name for name in ims_c3]
#
# target_size_seg = (256, 256)
# model_seg = unet_seg(input_size=target_size_seg + (1,))
# model_seg.load_weights(MODEL_FILE)
# # =================== seg part ===============
# segs = []
# for im_dir in ims_c1_dir:
#     segs.append(cell_segmentation(im_dir=im_dir, model_seg=model_seg))
# # =================== seg part ===============
# cells = []
# fovs_nm = len(segs)
# for seg_index in tqdm(range(fovs_nm)):
#     contours = cv2.findContours(segs[seg_index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # find
#     contours.sort(key=lambda elem: np.sum(np.max(elem[:, 0, 0] + np.max(elem[:, 0, 1]))))
#     im_c2 = tiff.imread(ims_c2_dir[seg_index]).astype(np.float)
#     im_c2_back_mean = np.mean(im_c2[np.logical_not(segs[seg_index])])  # c2 background
#     im_c2 = back_corrt(im_c2, im_c2_back_mean)
#     im_c3 = tiff.imread(ims_c3_dir[seg_index]).astype(np.float)
#     im_c3_back_mean = np.mean(im_c3[np.logical_not(segs[seg_index])])  # c3 background
#     im_c3 = back_corrt(im_c3, im_c3_back_mean)
#     for c, contour in enumerate(contours):  # Run through cells in frame
#         # Length, width, area:
#         cell_mask = np.zeros(segs[seg_index].shape, np.uint8)
#         cell_mask = cv2.drawContours(cell_mask, [contour], 0, 255, -1)
#         rotrect = cv2.minAreaRect(contour)
#         cell_pars = dict(
#             length=max(rotrect[1]),
#             width=min(rotrect[1]),
#             area=cv2.contourArea(contour),
#             c2_mean=cv2.mean(im_c2, cell_mask)[0],
#             c3_mean=cv2.mean(im_c3, cell_mask)[0])
#         cell = Cell(contour=contour, cell_paras=cell_pars, fov=seg_index)
#         cells.append(cell)
# Pixels list and Fluorescence:
# cellpixels = np.where(label_stack_resized[i] == cellnb)
# lineage[cellnb - 1]['pixels'].append(
#     np.ravel_multi_index(cellpixels, label_stack_resized.shape[1:]).astype(
#         np.float32))  # Using floats for compatibility with previous version of the pipeline
# for f in range(fluoframes.shape[1]):  # Go over all channels:
#     lineage[cellnb - 1]['fluo' + str(f + 1)].append(
#         np.mean(chamberfluo[f, cellpixels[0], cellpixels[1]]))

# # %%
# fig1, ax = plt.subplots(1, 1)
# cells_length = [c.cell_paras['length'] for c in cells]
# cells_width = [c.cell_paras['width'] for c in cells]
# ax.scatter(cells_width, cells_length)
# fig1.show()
# # %%
# fig2, ax = plt.subplots(1, 1)
# ax = sns.scatterplot(x=cells_width, y=cells_length)
# ax.set_xlim((0, 100))
# ax.set_ylim((0, 100))
# ax.plot(range(100), range(100))
# ax.set_xlabel('cell width')
# ax.set_ylabel('cell length')
# fig2.show()
# # %%
# cells_c2 = np.array([c.cell_paras['c2_mean'] for c in cells])
# cells_c3 = np.array([c.cell_paras['c3_mean'] for c in cells])
#
# fig3, ax = plt.subplots(1, 1)
# ax = sns.scatterplot(x=cells_c3, y=cells_c2)
# ax.set_ylim(1e-1, 1e5)
# ax.set_xlim(1e-1, 1e5)
# ax.set_ylabel('GFP')
# ax.set_xlabel('mCherry')
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig3.show()
# # %%
# fig4, ax = plt.subplots(1, 1)
# ax = plt.hist(x=np.log(cells_c3[cells_c3 > 0]), bins=100)
# plt.xlabel('log mCherry')
# fig4.show()
# %%

# im_phase = rangescale(tiff.imread(ims_c1_dir[seg_index]), (0, 255))
# im_contour = cv2.drawContours(im_phase, contours, -1, 255, 3)
# fig1, axes = plt.subplots(1, 1, figsize=(20, 20))
# axes.imshow(im_contour.astype(np.uint8))
# fig1.show()
#
# im_phase = rangescale(im_c3, (0, 255))
# im_phase[im_phase > 50] = 255
# im_contour = cv2.drawContours(im_phase, contours, -1, 255, 3)
# fig2, axes = plt.subplots(1, 1, figsize=(20, 20))
# axes.imshow(im_contour.astype(np.uint8))
# fig2.show()
#
# im_phase = rangescale(im_c2, (0, 255))
# im_phase[im_phase > 5] = 255
# im_contour = cv2.drawContours(im_phase, contours, -1, 255, 3)
# fig2, axes = plt.subplots(1, 1, figsize=(20, 20))
# axes.imshow(im_contour.astype(np.uint8))
# fig2.show()
#
# tiff.imsave(IMAGE_DIR + 'im_contour.tif', data=im_contour)
#
# # %%
#
# results = seg
# seg_results = seg_inputs
# fig4, ax4 = plt.subplots(1, results.shape[0], figsize=(18, 10))
# for i, ax in enumerate(ax4):
#     ax.imshow(results[i, ...])
# fig4.show()
#
# fig3, ax3 = plt.subplots(1, results.shape[0], figsize=(18, 10))
# for i, ax in enumerate(ax3):
#     ax.imshow(seg_results[i, ...].squeeze())
# fig3.show()
