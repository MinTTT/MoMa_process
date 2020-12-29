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
from utils.rotation import rotate_fov
from utils.signal import vertical_mean

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
process_size = 120
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
        file_name.sort(key=lambda elem: int(elem.split('.')[0][1:]))
        tim_dict[channel] = file_name
    return tim_dict


def get_im_time(ps):
    with tif.TiffFile(ps) as tim:
        return tim.asarray(), tim.shaped_metadata[0]['time']


def get_fluo_channel(ps, drift):
    im, time = get_im_time(ps)
    im, _ = driftcorr(img=im, template=None, box=None, drift=drift)
    return im, time


class MomoFov:
    def __init__(self, name, dir):
        self.fov_name = name
        self.dir = dir
        self.cell_minisize = 50
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
        self.chamber_cells_mask = dict()
        self.chamber_cells_contour = dict()
        self.loaded_chamber_name = []
        self.chamber_red_ims = dict()
        self.chamber_green_ims = dict()
        self.mother_cell_pars = dict()
        self.dataframe_mother_cells = pd.DataFrame(data=[])
        self.index_of_loaded_chamber = []

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
            self.chamberboxes = getChamberBoxes(np.squeeze(chambermask))
            self.drifttemplate = getDriftTemplate(self.chamberboxes, im.squeeze()[::-1, :])
        else:
            self.chamber_direction = 1
            self.drifttemplate = getDriftTemplate(self.chamberboxes, im.squeeze())
        return None

    def detect_frameshift(self):
        """
        This function used to detect frame shift.
        :return:
        """
        self.phase_ims = []
        phase_tp = []
        for fn in self.times['phase']:
            im, tp = get_im_time(os.path.join(self.dir, self.fov_name, 'phase', fn))
            if self.chamber_direction == 0:
                im = im[::-1, :]
            self.phase_ims.append(rangescale(im, (0, 1)))
            phase_tp.append(tp)
        self.time_points['phase'] = phase_tp
        self.phase_ims = np.array(self.phase_ims)
        print(f'ims shape is {self.phase_ims.shape}')
        driftcorbox = dict(xtl=0,
                           xbr=None,
                           ytl=0,
                           ybr=max(self.chamberboxes, key=lambda elem: elem['ytl'])['ytl']
                           )  # Box to match template
        self.phase_ims, self.drift_values = driftcorr(self.phase_ims,
                                                      template=self.drifttemplate, box=driftcorbox)
        xcorr_one = int(self.drift_values[0][0])
        ycorr_one = int(self.drift_values[1][0])
        for box in self.chamberboxes:
            box['ytl'] += xcorr_one
            box['ybr'] += xcorr_one
            box['xtl'] -= ycorr_one
            box['ytl'] -= ycorr_one
        # -------------detect whether chambers were loaded with cells--------
        num_time = len(self.times)
        sample_index = np.random.choice(range(num_time), int(num_time * 0.01 + 1))
        selected_ims = self.phase_ims[sample_index, ...]
        cells_threshold = 0.23
        chamber_loaded = []
        for box in self.chamberboxes:
            mean_chamber = np.mean(selected_ims[:, box['ytl']:box['ybr'], box['xtl']:box['xbr']])
            if mean_chamber < cells_threshold:
                chamber_loaded.append(True)
            else:
                chamber_loaded.append(False)
        self.index_of_loaded_chamber = list(np.where(chamber_loaded)[0])
        self.loaded_chamber_box = [self.chamberboxes[index] for index in self.index_of_loaded_chamber]

    def cell_detection(self):
        seg_inputs = []
        # Compile segmentation inputs:
        for m, chamberbox in enumerate(self.loaded_chamber_box):
            for i in range(self.phase_ims.shape[0]):
                seg_inputs.append(cv2.resize(rangescale(cropbox(self.phase_ims[i], chamberbox), (0, 1)), (32, 256)))

        seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)
        # Format into 4D tensor
        # Run segmentation U-Net:
        seg = model_seg.predict(seg_inputs, verbose=1)
        self.cell_mask = postprocess(seg[:, :, :, 0], min_size=self.cell_minisize)
        # -------------- reform the size--------------
        for m, box in enumerate(self.loaded_chamber_box):
            ori_frames = np.empty([len(self.times['phase']), box['ybr'] - box['ytl'], box['xbr'] - box['xtl']]).astype(
                np.uint16)
            for t in range(len(self.times['phase'])):
                frame_index = t + m * len(self.times['phase'])
                ori_frames[t] = cv2.resize(self.cell_mask[frame_index], ori_frames.shape[2:0:-1])
            self.chamber_cells_mask[f'ch_{m}'] = ori_frames
        # -------------- get cells contour ------------
        self.loaded_chamber_name = list(self.chamber_cells_mask.keys())
        for m, channel in enumerate(self.loaded_chamber_name):
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
        for inx_t, time in enumerate(self.times['phase']):
            """get all fluorescent images from disk and seg into channel XX_channels is a dictionary keys are file 
            names and their elements are lists containing chambers ordered by loaded chamber in 
            self.loaded_chamber_name. 
            """
            if 'green' in self.channels:
                green_time_points = []
                if time in self.times['green']:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    green_im, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, 'green', time),
                                                            drift_valu)
                    green_channels[time] = [cropbox(green_im, cb) for cb in self.loaded_chamber_box]
                    green_time_points.append(time_point)
            if 'red' in self.channels:
                red_time_points = []
                if time in self.times['red']:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    red_im, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, 'red', time),
                                                          drift_valu)
                    red_channels[time] = [cropbox(red_im, cb) for cb in self.loaded_chamber_box]
                    red_time_points.append(time_point)
            self.time_points['green'] = green_time_points
            self.time_points['red'] = red_time_points

        for cha_name_inx, chambername in enumerate(self.loaded_chamber_name):
            self.mother_cell_pars[chambername] = []
            for tm_inx, time in enumerate(self.times['phase']):
                cell_mask = np.zeros(self.chamber_cells_mask[chambername].shape[1:], np.uint8)
                if len(self.chamber_cells_contour[chambername][tm_inx]) != 0:
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
                        cell_pars['green_mean'] = cv2.mean(green_channel_im, cell_mask)
                    if time in self.times['red']:
                        red_channel_im = red_channels[time][cha_name_inx]
                        cell_pars['red_mean'] = cv2.mean(red_channel_im, cell_mask)
                    self.mother_cell_pars[chambername].append(cell_pars)
        # This section was used to segment the single chamber frame along time.
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
        pf_list = []
        for chna in self.loaded_chamber_name:
            pf = pd.DataFrame(data=self.mother_cell_pars[chna])
            time = [float(t.split(',')[-1]) for t in pf['time_point']]
            pf['time_s'] = time
            pf['channel'] = [chna] * len(pf)
            pf_list.append(pf)
        self.dataframe_mother_cells = pd.concat(pf_list, sort=False)
        self.dataframe_mother_cells.index = pd.Index(range(len(self.dataframe_mother_cells)))

    def process_flow(self):
        print('Now, detect channels.\n')
        self.detect_channels()
        print('Now, detect frameshift.\n')
        self.detect_frameshift()
        print('Now, detect cells.\n')
        self.cell_detection()
        print("Now, extract cells' features.\n")
        self.extract_mother_cells_features()
        print("Now, get mother cells data.\n")
        self.parse_mother_cell_data()

# %%
DIR = r'F:\ZJW_CP\20201227'
fovs_name = [MomoFov(folder, DIR) for folder in os.listdir(DIR) if folder.split('_')[0] == 'fov']

fovs_name[1].process_flow()

# %%
fig1, ax = plt.subplots(1, 1)
ax.imshow(fovs_name[0].chamber_cells_mask['ch_11'][8])
fig1.show()
# %% make sure their cells in channel
num_time = len(fovs_name[0].times)
sample_index = np.random.choice(range(num_time), int(num_time * 0.01 + 1))
selected_ims = fovs_name[0].phase_ims[sample_index, ...]
cells_threshold = 0.23
chamber_loaded = []
for box in fovs_name[0].chamberboxes:
    mean_chamber = np.mean(selected_ims[:, box['ytl']:box['ybr'], box['xtl']:box['xbr']])
    if mean_chamber < cells_threshold:
        chamber_loaded.append(True)
    else:
        chamber_loaded.append(False)
loaded_chamber_box = [fovs_name[0].chamberboxes[index] for index in list(np.where(chamber_loaded)[0])]

# %%
fovs_name = [folder for folder in os.listdir(DIR) if folder.split('_')[0] == 'fov']
flu_channels_name = os.listdir(os.path.join(DIR, fovs_name[0]))
# time_points = os.listdir(os.path.join(DIR, fovs_name[0], flu_channels_name[0]))
fovs_list = []
for f, fov in enumerate(tqdm(fovs_name)):
    flu_channels_name = os.listdir(os.path.join(DIR, fovs_name[f]))
    times = []
    for flu, flu_ch in enumerate(flu_channels_name):
        time_points = os.listdir(os.path.join(DIR, fovs_name[f], flu_channels_name[flu]))
        times.append(time_points)
    fovs_list.append(times)
# %%
TEST_SEG_FOV = fovs_list[1][1]

# im = tif.imread(os.path.join(DIR, fovs_name[1], flu_channels_name[1], TEST_SEG_FOV[0]))
# v_m = vertical_mean(im)
# v_m[(v_m - np.median(v_m)) < 0] = 0
# f_v = np.mean(v_m[0:len(v_m)//2])
# r_v = np.mean(v_m[len(v_m)//2:-1])
ims = []
for im_name in tqdm(fovs_list[1][1][0:10]):
    im = io.imread(os.path.join(DIR, fovs_name[1], flu_channels_name[1], im_name))
    im, _ = rotate_fov(np.expand_dims(im, axis=0), crop=False)
    im = rangescale(im.squeeze(), rescale=(0, 1))
    ims.append(im)
ims = np.array(ims)
# im_crp = cropbox(im_ro.squeeze(), roi)
# im_crp = im_crp[::-1, ...]
# io.imsave('/'.join([save_folder, im_name]), im_crp, check_contrast=False)

ims = ims[:, ::-1, :]

# Rotation correction:
firstframe = np.expand_dims(np.expand_dims(cv2.resize(ims[0].squeeze(), (512, 512)), axis=0), axis=3)
# using expand_dims to get it into a shape that the chambers id unet accepts
# Find chambers, filter results, get bounding boxes:
chambermask = model_chambers.predict(firstframe, verbose=0)
chambermask = cv2.resize(np.squeeze(chambermask), ims.shape[3:0:-1])  # scaling back to original size
chambermask = postprocess(chambermask, min_size=min_chamber_area)  # Binarization, cleaning and area filtering
chamberboxes = getChamberBoxes(np.squeeze(chambermask))

# Drift correction:
drifttemplate = getDriftTemplate(chamberboxes,
                                 ims[0].squeeze())  # This template will be used as reference for drift correction
driftcorbox = dict(xtl=0,
                   xbr=None,
                   ytl=0,
                   ybr=max(chamberboxes, key=lambda elem: elem['ytl'])['ytl']
                   )  # Box to match template
ims, driftvalues = driftcorr(ims, template=drifttemplate, box=driftcorbox)  # Run drift corr
# Load up fluoresence images, apply drift correction and rotation:


##### Cell segmentation:

seg_inputs = []
# Compile segmentation inputs:
for m, chamberbox in enumerate(chamberboxes):
    for i in range(ims.shape[0]):
        seg_inputs.append(cv2.resize(rangescale(cropbox(ims[i], chamberbox), (0, 1)),
                                     (32, 256)))

seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)
# Format into 4D tensor
# Run segmentation U-Net:
seg = model_seg.predict(seg_inputs, verbose=1)
seg = postprocess(seg[:, :, :, 0])
