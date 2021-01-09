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
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import asyncio

import time
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
from joblib import Parallel, load, dump, delayed
from utils.delta.utilities import cropbox
from utils.rotation import rotate_fov, rotate_image
from utils.signal import vertical_mean
import dask
# dask.config.set(pool=ThreadPool(64))
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar

client = Client(threads_per_worker=64, n_workers=2)

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2 * 2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

model_file = r'test_data_set/model/chambers_id_tessiechamp_old.hdf5'
seg_model_file = r'./test_data_set/model/unet_moma_seg_multisets.hdf5'
min_chamber_area = 3e3
target_size = (512, 512)
input_size = target_size + (1,)
process_size = 200
model_chambers = unet_chambers(input_size=input_size)
model_chambers.load_weights(model_file)
target_size_seg = (256, 32)
model_seg = unet_seg(input_size=target_size_seg + (1,))
model_seg.load_weights(seg_model_file)


def get_channel_name(dir, name):
    return os.listdir(os.path.join(dir, name))


def get_times(dir, name, channels):
    tim_dict = dict()
    for channel in channels:
        file_name = os.listdir(os.path.join(dir, name, channel))
        file_name = [name for name in file_name if name.split('.')[-1] == 'tiff']
        file_name.sort(key=lambda elem: int(elem.split('.')[0][1:]))
        tim_dict[channel] = file_name
    return tim_dict


def crop_images(imgs, box):
    """
    Crop images

    Parameters
    ----------
    imgs : 3D numpy array [t, x, y]
        Image to crop.
    box : Dictionary
        Dictionary describing the box to cut out, containing the following
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    """
    return imgs[:, box['ytl']:box['ybr'], box['xtl']:box['xbr']]


def back_corrt(im: np.ndarray) -> np.ndarray:
    im -= int(np.median(im))
    im[im < 0] = 0.
    return im


def get_im_time(ps):
    with tif.TiffFile(ps) as tim:
        return tim.asarray(), tim.shaped_metadata[0]['time']


def get_fluo_channel(ps, drift, angle):
    im, time = get_im_time(ps)
    im = rotate_image(im, angle)
    im, _ = driftcorr(img=im, template=None, box=None, drift=drift)
    return im, time


def parallel_seg_input(ims, box, size=(256, 32)):
    """
    seg images and resized to a certain size.
    :param ims: frames
    :param box: chamberbox
    :param size: resize scale
    :return: resize images, un-resized images
    """

    def resize_map(i, size):
        resize_ims[i, ...] = cv2.resize(subims[i, ...], size[::-1])
        return None

    subims = ims[:, box['ytl']:box['ybr'], box['xtl']:box['xbr']]
    ims_num = len(subims)
    resize_ims = np.empty((ims_num,) + size)
    _ = Parallel(n_jobs=30, require='sharedmem')(delayed(resize_map)(im_inx, size) for im_inx in range(ims_num))
    return resize_ims, subims


class MomoFov:
    def __init__(self, name, dir):
        """
        :param name: str, fov name eg. 'fov_0'
        :param dir: str, path of fovs
        """
        self.fov_name = name
        self.dir = dir
        self.cell_minisize = 100
        self.channels = get_channel_name(self.dir, self.fov_name)
        self.times = None
        self.phase_ims = None
        self.time_points = dict()
        self.chambermask = None
        self.chamberboxes = []
        self.loaded_chamber_box = []
        self.drifttemplate = None
        self.template_frame = None
        self.chamber_direction = None
        self.drift_values = None
        self.cell_mask = None  # NOTE: cell_mask have a order which is dependent on times. i.e. #chan_1 ---times----
        # #chan_2 --- times---
        self.chamber_cells_mask = dict()  # resized cells mask
        self.chamber_cells_contour = dict()
        self.loaded_chamber_name = []
        self.chamber_red_ims = dict()
        self.chamber_green_ims = dict()
        self.chamber_phase_ims = dict()
        self.mother_cell_pars = dict()
        self.dataframe_mother_cells = pd.DataFrame(data=[])
        self.index_of_loaded_chamber = []
        self.chamber_graylevel = []
        self.chamber_seg = None
        self.rotation = []

    def detect_channels(self, number=0):
        """
        The frame number used to detect side_channels. default is 0
        :param number: int,
        :return:
        """
        self.times = get_times(self.dir, self.fov_name, self.channels)
        im, _ = get_im_time(os.path.join(self.dir, self.fov_name, 'phase', self.times['phase'][0]))
        im, _ = rotate_fov(np.expand_dims(im, axis=0), crop=False)
        im = rangescale(im.squeeze(), rescale=(0, 1))
        self.template_frame = im
        firstframe = np.expand_dims(np.expand_dims(cv2.resize(im.squeeze(), (512, 512)), axis=0), axis=3)
        # using expand_dims to get it into a shape that the chambers id unet accepts
        # Find chambers, filter results, get bounding boxes:
        chambermask = model_chambers.predict(firstframe, verbose=0)
        chambermask = cv2.resize(np.squeeze(chambermask), im.shape)  # scaling back to original size
        chambermask = postprocess(chambermask, min_size=min_chamber_area)  # Binarization, cleaning and area filtering
        self.chamberboxes = getChamberBoxes(np.squeeze(chambermask))

        print(f"{self.fov_name}: detect {len(self.chamberboxes)} chambers.")
        border = int(im.shape[0] * 0.02)
        chambercorbox = dict(xtl=min(self.chamberboxes, key=lambda elem: elem['xtl'])['xtl'],
                             xbr=max(self.chamberboxes, key=lambda elem: elem['xbr'])['xbr'],
                             ytl=min(self.chamberboxes, key=lambda elem: elem['ytl'])['ytl'] - border,
                             ybr=max(self.chamberboxes, key=lambda elem: elem['ybr'])['ybr'] + border
                             )
        v_m = vertical_mean(cropbox(self.template_frame, chambercorbox))
        if np.mean(v_m[0:(len(v_m) // 2)]) >= np.mean(v_m[(len(v_m) // 2):]):
            self.chamber_direction = 0
            self.chambermask = chambermask[::-1, :]
            self.chamberboxes = getChamberBoxes(self.chambermask)
            self.drifttemplate = getDriftTemplate(self.chamberboxes, im.squeeze()[::-1, :])
        else:
            self.chamber_direction = 1
            self.chambermask = chambermask
            self.drifttemplate = getDriftTemplate(self.chamberboxes, im.squeeze())
        return None

    def detect_frameshift(self):
        """
        This function used to detect frame shift.
        :return:
        """
        print(f'{self.fov_name}:loading phase images. \n')
        self.phase_ims = np.zeros((len(self.times['phase']),) + self.template_frame.shape)
        self.time_points['phase'] = [False] * len(self.times['phase'])
        self.rotation = [None] * len(self.times['phase'])

        def parallel_input(fn, inx):
            im, tp = get_im_time(os.path.join(self.dir, self.fov_name, 'phase', fn))
            if self.chamber_direction == 0:
                im = im[::-1, :]
            im, angl = rotate_fov(np.expand_dims(im, axis=0), crop=False)
            self.phase_ims[inx, ...] = rangescale(im.squeeze(), (0, 1))
            self.time_points['phase'][inx] = tp
            self.rotation[inx] = angl
            return None
        # --------------------- input all phase images --------------------------------------
        _ = Parallel(n_jobs=30, require='sharedmem')(
            delayed(parallel_input)(fn, i) for i, fn in tqdm(enumerate(self.times['phase'])))

        print(f'{self.fov_name}: ims shape is {self.phase_ims.shape}.')
        # --------------------- input all phase images --------------------------------------
        driftcorbox = dict(xtl=0,
                           xbr=None,
                           ytl=0,
                           ybr=max(self.chamberboxes, key=lambda elem: elem['ytl'])['ytl']
                           )  # Box to match template
        self.phase_ims, self.drift_values = driftcorr(self.phase_ims,
                                                      template=self.drifttemplate, box=driftcorbox)
        xcorr_one = int(self.drift_values[0][0])
        ycorr_one = int(self.drift_values[1][0])
        for box in self.chamberboxes:  # TODO: frame shift have bug.
            box['ytl'] += xcorr_one
            box['ybr'] += xcorr_one
            box['xtl'] -= ycorr_one
            box['ytl'] -= ycorr_one
        # -------------detect whether chambers were loaded with cells--------
        num_time = len(self.times['phase'])
        sample_index = np.random.choice(range(num_time), int(num_time * 0.01 + 1))
        selected_ims = self.phase_ims[sample_index, ...]

        chamber_graylevel = []
        for box in self.chamberboxes:
            half_chambers = selected_ims[:, box['ytl']:int((box['ybr'] - box['ytl']) / 2 + box['ytl']),
                            box['xtl']:box['xbr']]
            mean_chamber = np.mean(half_chambers)
            chamber_graylevel.append(mean_chamber)
        cells_threshold = np.min(chamber_graylevel) + np.ptp(chamber_graylevel) * 0.6
        chamber_loaded = [True if value < cells_threshold else False for value in chamber_graylevel]
        self.index_of_loaded_chamber = list(np.where(chamber_loaded)[0])
        self.loaded_chamber_box = [self.chamberboxes[index] for index in self.index_of_loaded_chamber]
        self.chamber_graylevel = chamber_graylevel
        print(chamber_graylevel)
        print(f'{self.fov_name}: detect chamber loaded number: {len(self.loaded_chamber_box)}.')

    def cell_detection(self):
        seg_inputs = ()
        for m, chamberbox in enumerate(self.loaded_chamber_box):
            if chamberbox:
                sub_inputs, ori_chn_imgs = parallel_seg_input(self.phase_ims, chamberbox)
                seg_inputs += (sub_inputs,)
                self.chamber_phase_ims[f'ch_{self.index_of_loaded_chamber[m]}'] = ori_chn_imgs
        seg_inputs = np.concatenate(seg_inputs, axis=0)
        self.phase_ims = None  # release memory
        seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)
        self.chamber_seg = seg_inputs
        # Format into 4D tensor
        # Run segmentation U-Net:
        seg = model_seg.predict(seg_inputs, verbose=1)
        self.cell_mask = postprocess(seg[:, :, :, 0], min_size=self.cell_minisize)

        # -------------- reform the size-------------- TODO: parallel 1
        def parallel_rearange_mask(t, chn):
            frame_index = t + chn * len(self.times['phase'])
            ori_frames[t] = cv2.resize(self.cell_mask[frame_index], ori_frames.shape[2:0:-1])

        for m, box in enumerate(self.loaded_chamber_box):
            ori_frames = np.empty([len(self.times['phase']), box['ybr'] - box['ytl'], box['xbr'] - box['xtl']]).astype(
                np.uint16)

            rerange_mask = [dask.delayed(parallel_rearange_mask)(t, m) for t in range(len(self.times['phase']))]
            _ = dask.compute(*rerange_mask, scheduler='threads')
            # for t in range(len(self.times['phase'])):
            #     frame_index = t + m * len(self.times['phase'])
            #     ori_frames[t] = cv2.resize(self.cell_mask[frame_index], ori_frames.shape[2:0:-1])

            self.chamber_cells_mask[f'ch_{self.index_of_loaded_chamber[m]}'] = ori_frames
        # -------------- get cells contour ------------
        self.loaded_chamber_name = list(self.chamber_cells_mask.keys())
        for channel in self.loaded_chamber_name:
            contours_list = []
            for time_mask in self.chamber_cells_mask[channel]:
                contours = cv2.findContours((time_mask > 0).astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0]  # Get contours of single cells
                contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sort along Y axis
                contours_list.append(contours)
            self.chamber_cells_contour[channel] = contours_list

    def extract_mother_cells_features(self):
        green_channels = dict()
        red_channels = dict()
        green_time_points = dict()
        red_time_points = dict()

        def parallel_flur_seg(inx_t, time):
            """
            get all fluorescent images from disk and seg into channel XX_channels is a dictionary keys are file 
            names and their elements are lists containing chambers ordered by loaded chamber in 
            self.loaded_chamber_name. 
            """
            if 'green' in self.channels:
                if time in self.times['green']:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    green_im, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, 'green', time),
                                                            drift_valu, self.rotation[inx_t][0])
                    green_im = back_corrt(green_im.astype(np.float64))  # fluorescence background correct
                    if self.chamber_direction == 0:
                        green_im = green_im[::-1, :]
                    green_channels[time] = [cropbox(green_im, cb) for cb in self.loaded_chamber_box]
                    green_time_points[time] = time_point

            if 'red' in self.channels:
                if time in self.times['red']:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    red_im, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, 'red', time),
                                                          drift_valu, self.rotation[inx_t][0])
                    red_im = back_corrt(red_im.astype(np.float64))
                    if self.chamber_direction == 0:
                        red_im = red_im[::-1, :]
                    red_channels[time] = [cropbox(red_im, cb) for cb in self.loaded_chamber_box]
                    red_time_points[time] = time_point
            return time

        parall_results = [dask.delayed(parallel_flur_seg)(inx_t, time) for inx_t, time in
                          enumerate(self.times['phase'])]
        print(f'Now, {self.fov_name}: loading fluorescent images.')
        with ProgressBar():
            _ = dask.compute(*parall_results, scheduler='threads')

        self.time_points['green'] = [green_time_points[i] for i in self.times['green']]
        self.time_points['red'] = [red_time_points[i] for i in self.times['red']]

        for cha_name_inx, chambername in enumerate(self.loaded_chamber_name):
            self.mother_cell_pars[chambername] = []
            for tm_inx, time in enumerate(self.times['phase']):
                cell_mask = np.zeros(self.chamber_cells_mask[chambername].shape[1:], np.uint8)
                if self.chamber_cells_contour[chambername][tm_inx]:
                    mother_cell_contour = self.chamber_cells_contour[chambername][tm_inx][0]
                    cell_mask = cv2.drawContours(cell_mask, [mother_cell_contour], 0, 255, -1)
                    rotrect = cv2.minAreaRect(mother_cell_contour)
                    cell_pars = dict(
                        length=max(rotrect[1]),
                        width=min(rotrect[1]),
                        area=cv2.contourArea(mother_cell_contour),
                        time_point=self.time_points['phase'][tm_inx]
                    )
                    if time in self.times['green']:
                        green_channel_im = green_channels[time][cha_name_inx]
                        green_pixels = green_channel_im[cell_mask == 255]
                        # cell_pars['green_mean'] = cv2.mean(green_channel_im, cell_mask)[0]
                        cell_pars['green_mean'] = np.mean(green_pixels)
                        # medium only consider the medium of the brightest 10% pixels.
                        cell_pars['green_medium'] = np.quantile(green_pixels, 0.95)
                    if time in self.times['red']:
                        red_channel_im = red_channels[time][cha_name_inx]
                        red_pixels = red_channel_im[cell_mask == 255]
                        # cell_pars['red_mean'] = cv2.mean(red_channel_im, cell_mask)[0]
                        cell_pars['red_mean'] = np.mean(red_pixels)
                        # medium only consider the medium of the brightest 10% pixels.
                        cell_pars['red_medium'] = np.quantile(red_pixels, 0.95)
                    self.mother_cell_pars[chambername].append(cell_pars)
        # This section was used to rearrange the single chamber frame along time.
        if 'green' in self.channels:
            for chamber_index, cham_name in enumerate(self.loaded_chamber_name):
                imgsize = (len(self.times['green']),) + green_channels[self.times['green'][0]][chamber_index].shape
                green_chamber_ims = np.empty(imgsize).astype(np.uint16)
                for t_index, time in enumerate(self.times['green']):
                    green_chamber_ims[t_index] = green_channels[time][chamber_index]
                self.chamber_green_ims[cham_name] = green_chamber_ims
        if 'red' in self.channels:
            for chamber_index, cham_name in enumerate(self.loaded_chamber_name):
                imgsize = (len(self.times['red']),) + red_channels[self.times['red'][0]][chamber_index].shape
                red_chamber_ims = np.empty(imgsize).astype(np.uint16)
                for t_index, time in enumerate(self.times['red']):
                    red_chamber_ims[t_index] = red_channels[time][chamber_index]
                self.chamber_red_ims[cham_name] = red_chamber_ims

    def parse_mother_cell_data(self):
        """
        parse all statistics data into one frame.
        :return:
        """
        pf_list = []
        for chna in self.loaded_chamber_name:  # drop channels don't have cells.
            if self.mother_cell_pars[chna]:
                pf = pd.DataFrame(data=self.mother_cell_pars[chna])
                time = [float(t.split(',')[-1]) for t in pf['time_point']]
                pf['time_s'] = time
                pf['chamber'] = [f'{self.fov_name}_{chna}'] * len(pf)
                pf_list.append(pf)
        self.dataframe_mother_cells = pd.concat(pf_list, sort=False)
        self.dataframe_mother_cells.index = pd.Index(range(len(self.dataframe_mother_cells)))
        return None

    def dump_data(self):
        self.dataframe_mother_cells.to_csv(os.path.join(self.dir, self.fov_name + '_statistic.csv'))
        save_data = dict(directory=self.dir,
                         fov_name=self.fov_name,
                         frame_rotation_anle=self.rotation,
                         frame_shift=self.drift_values,
                         times=self.times,
                         time_points=self.time_points,
                         light_channels=self.channels,
                         chamber_box=self.chamberboxes,
                         chamber_loaded_name=self.loaded_chamber_name,
                         chamber_loaded_index=self.index_of_loaded_chamber,
                         chamber_grayvalue=self.chamber_graylevel,
                         chamber_direction=self.chamber_direction,
                         chamber_cells_mask=self.chamber_cells_mask,
                         chamber_cells_contour=self.chamber_cells_contour,
                         chamber_phase_images=self.chamber_phase_ims,
                         chamber_green_images=self.chamber_green_ims,
                         chamber_red_images=self.chamber_red_ims,
                         mother_cells_parameters=self.mother_cell_pars,
                         mother_cells_dataframe=self.dataframe_mother_cells
                         )
        dump(save_data, os.path.join(self.dir, self.fov_name + '.jl'), compress='lzma')
        return None

    def process_flow(self):
        print(f'Now, {self.fov_name}: detect channels.\n')
        self.detect_channels()
        print(f'Now, {self.fov_name}: detect frameshift.\n')
        self.detect_frameshift()
        print(f'Now, {self.fov_name}: detect cells.\n')
        self.cell_detection()
        print(f"Now, {self.fov_name}: extract cells' features.\n")
        self.extract_mother_cells_features()
        print(f"Now, {self.fov_name}: get mother cells data.\n")
        self.parse_mother_cell_data()
        print(f"Now, {self.fov_name}: dump memory data.\n")
        self.dump_data()
        return None


def get_fovs_name(dir, all_fov=False):
    """
    Get fovs name under dir, if fovs in dir have been treated (i.e. having a memory obj in folder dir), these fovs will
    not returned.
    :param dir: str, ps
    :param all_fov: bool, if True, return all folders, default, False.
    :return: list
    """
    DIR = dir
    jl_file = [jl_name.split('.')[0] for jl_name in os.listdir(DIR) if jl_name.split('.')[-1] == 'jl']
    fov_folder = [folder for folder in os.listdir(DIR)
                  if (folder.split('_')[0] == 'fov' and os.path.isdir(os.path.join(DIR, folder)))]
    if all_fov == False:
        untreated = list(set(fov_folder) - set(jl_file))
        untreated.sort(key=lambda name: int(name.split('_')[-1]))
        fovs_name = [MomoFov(folder, DIR) for folder in untreated]
        return fovs_name
    else:
        fov_folder.sort(key=lambda name: int(name.split('_')[-1]))
        return fov_folder


# %%
if __name__ == '__main__':
    DIR = r'X:\chupan\mother machine\20201221_NH2_PECJ3'

    untreated = get_fovs_name(DIR)

    for i, fov in enumerate(untreated):
        print(f'Processing {i + 1}/{len(untreated)}')
        fov.process_flow()
        del untreated[i]

# %%
#
# jldb = load(os.path.join(DIR, 'fov_90.jl'))
#
# # %%
# print(jldb.loaded_chamber_name)
# fig1, ax = plt.subplots(1, 1)
# im = jldb.chamber_cells_mask['ch_8'][18]
# print(np.mean(im))
# ax.imshow(im)
# fig1.show()
# # %%
#
# num_time = len(fov.time_points)
# sample_index = np.random.choice(range(num_time), int(num_time * 0.01 + 1))
# selected_ims = jldb.phase_ims[sample_index, ...]
# cells_threshold = 0.30
# chamber_loaded = []
# for box in jldb.chamberboxes:
#     half_chambers = selected_ims[:, box['ytl']:int(box['ybr'] - box['ytl'] / 2 + box['ytl']), box['xtl']:box['xbr']]
#     mean_chamber = np.mean(half_chambers)
#     print(mean_chamber)
