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

import time
# […]

# Own modules
from utils.delta.data import postprocess
from utils.delta.model import unet_chambers, unet_seg
import numpy as np
import tensorflow as tf
import cv2
import tifffile as tif
from utils.delta.utilities import getChamberBoxes, getDriftTemplate, driftcorr, rangescale, cropbox
from joblib import Parallel, dump, delayed
from utils.rotation import rotate_fov, rotate_image
from utils.signal import vertical_mean
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.fftpack import fft, fftfreq
# import dask
# # dask.config.set(pool=ThreadPool(64))
# from dask.distributed import Client, progress
# from dask.diagnostics import ProgressBar
# client = Client(threads_per_worker=64, n_workers=1)

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

colors_2 = ['#FFA2A8', '#95FF57']  # red, green


def get_channel_name(dir, name):
    return os.listdir(os.path.join(dir, name))


def box_2_pltrec(box):
    plt_x, plt_y = box['xtl'], box['ytl']
    h = int(box['ybr'] - box['ytl'])
    w = int(box['xbr'] - box['xtl'])
    return dict(xy=(plt_x, plt_y), width=w, height=h)


def draw_channel_order(im, chn_boxs, colors, ax=None):
    if ax == None:
        ax = plt.gca()
    ax.imshow(im, cmap='gray')
    for i, box in enumerate(chn_boxs):
        ax.add_patch(Rectangle(**box_2_pltrec(box), fill=False, edgecolor=colors[i]))
    return ax


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


def get_fluo_channel(ps, drift, angle, direct):
    im, time = get_im_time(ps)
    im = rotate_image(im, angle)
    if direct:
        im, _ = driftcorr(img=im, template=None, box=None, drift=drift)
    else:
        im, _ = driftcorr(img=im[::-1, :], template=None, box=None, drift=drift)
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
        self.dataframe_mother_cells = None
        self.index_of_loaded_chamber = []
        self.chamber_graylevel = []
        self.chamber_seg = None
        self.rotation = []

    def detect_channels(self, number=0):
        """
        The frame number used to detect side_channels. default is 0
        images inputted and rotated than vertical flip if needed, than detect the fame shift
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

        print(f"[{self.fov_name}] -> detect {len(self.chamberboxes)} chambers.")
        border = int(im.shape[0] * 0.02)
        chambercorbox = dict(xtl=min(self.chamberboxes, key=lambda elem: elem['xtl'])['xtl'],
                             xbr=max(self.chamberboxes, key=lambda elem: elem['xbr'])['xbr'],
                             ytl=min(self.chamberboxes, key=lambda elem: elem['ytl'])['ytl'] - border,
                             ybr=max(self.chamberboxes, key=lambda elem: elem['ybr'])['ybr'] + border
                             )
        v_m = vertical_mean(cropbox(self.template_frame, chambercorbox))

        # image vertical flip if chamber out let in the upside.
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
        print(f'[{self.fov_name}] -> loading phase images. \n')
        self.phase_ims = np.zeros((len(self.times['phase']),) + self.template_frame.shape)
        self.time_points['phase'] = [False] * len(self.times['phase'])
        self.rotation = [None] * len(self.times['phase'])

        def parallel_input(fn, inx):
            im, tp = get_im_time(os.path.join(self.dir, self.fov_name, 'phase', fn))
            im, angl = rotate_fov(np.expand_dims(im, axis=0), crop=False)
            if self.chamber_direction == 0:
                im = im[:, ::-1, :]
            self.phase_ims[inx, ...] = rangescale(im.squeeze(), (0, 1))
            self.time_points['phase'][inx] = tp
            self.rotation[inx] = angl
            return None

        # --------------------- input all phase images --------------------------------------
        _ = Parallel(n_jobs=30, require='sharedmem')(
            delayed(parallel_input)(fn, i) for i, fn in enumerate(tqdm(self.times['phase'])))

        print(f'[{self.fov_name}] -> ims shape is {self.phase_ims.shape}.')
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
        chamber_frq = []

        for box in self.chamberboxes:
            half_chambers = selected_ims[:, box['ytl']:int((box['ybr'] - box['ytl']) / 2 + box['ytl']),
                            box['xtl']:box['xbr']]
            mean_chamber = np.mean(half_chambers)
            chamber_graylevel.append(mean_chamber)
            sg, frq = image_conv(selected_ims[:, int(box['ytl'] + (box['ybr'] - box['ytl']) * 0.15):int(
                (box['ybr'] - box['ytl']) * 0.5 + box['ytl']),
                                 box['xtl']:box['xbr']])
            chamber_frq.append([sg, frq])

        cells_threshold = np.min(chamber_graylevel) + np.ptp(chamber_graylevel) * 0.6
        chamber_loaded = [True if value < cells_threshold else False for value in chamber_graylevel]
        self.index_of_loaded_chamber = list(np.where(chamber_loaded)[0])
        self.loaded_chamber_box = [self.chamberboxes[index] for index in self.index_of_loaded_chamber]

        fig_sg, ax = plt.subplots(1, 1)
        for i in range(len(chamber_frq)):
            sg, frq = chamber_frq[i]
            if i in self.index_of_loaded_chamber:
                ax.plot(frq, np.log(abs(sg)), '--k', alpha=0.2)
            else:
                ax.plot(frq, np.log(abs(sg)), '-r')

        try:
            fig_sg_ps = os.path.join(self.dir, 'chamber_load')
            os.makedirs(fig_sg_ps)
        except FileExistsError:
            pass

        fig_sg.savefig(os.path.join(fig_sg_ps, f'{self.fov_name}_F.svg'), format='svg')

        fig_ch, ax = plt.subplots(1, 1)
        colors = [colors_2[1] if i in self.index_of_loaded_chamber else colors_2[0]
                  for i in range(len(chamber_frq))]
        draw_channel_order(rangescale(self.phase_ims[0, ...], (0, 255)).astype(np.uint8),
                           self.chamberboxes, colors, ax=ax)
        fig_ch.savefig(os.path.join(fig_sg_ps, f'{self.fov_name}_chamber.svg'), format='svg')

        self.chamber_graylevel = chamber_graylevel
        # print(f'[{self.fov_name}] -> , chamber_graylevel)
        print(f'[{self.fov_name}] -> detect loaded chamber number: {len(self.loaded_chamber_box)}.')
        return [self.index_of_loaded_chamber, self.chamber_graylevel, chamber_frq]

    def cell_detection(self):
        seg_inputs = ()
        for m, chamberbox in enumerate(self.loaded_chamber_box):
            if chamberbox:
                sub_inputs, ori_chn_imgs = parallel_seg_input(self.phase_ims, chamberbox)
                seg_inputs += (sub_inputs,)
                self.chamber_phase_ims[f'ch_{self.index_of_loaded_chamber[m]}'] = ori_chn_imgs
        seg_inputs = np.concatenate(seg_inputs, axis=0)
        del self.phase_ims  # release memory
        seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)
        self.chamber_seg = seg_inputs
        # Format into 4D tensor
        # Run segmentation U-Net:
        seg = model_seg.predict(seg_inputs)
        self.cell_mask = postprocess(seg[:, :, :, 0], min_size=self.cell_minisize)

        # -------------- reform the size--------------
        def parallel_rearange_mask(t, chn):
            frame_index = t + chn * len(self.times['phase'])
            ori_frames[t] = cv2.resize(self.cell_mask[frame_index], ori_frames.shape[2:0:-1])

        for m, box in enumerate(self.loaded_chamber_box):
            ori_frames = np.empty([len(self.times['phase']), box['ybr'] - box['ytl'], box['xbr'] - box['xtl']]).astype(
                np.uint16)

            # rerange_mask = [dask.delayed(parallel_rearange_mask)(t, m) for t in range(len(self.times['phase']))]
            # _ = dask.compute(*rerange_mask)

            _ = Parallel(n_jobs=64, require='sharedmem')(delayed(parallel_rearange_mask)(t, m)
                                                         for t in range(len(self.times['phase'])))

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
        # TODO: bug: when channel have no cells, the features results are strange.
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
                                                            drift_valu, self.rotation[inx_t][0],
                                                            self.chamber_direction)
                    green_im = back_corrt(green_im.astype(np.float64))  # fluorescence background correct
                    green_channels[time] = [cropbox(green_im, cb) for cb in self.loaded_chamber_box]
                    green_time_points[time] = time_point

            if 'red' in self.channels:
                if time in self.times['red']:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    red_im, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, 'red', time),
                                                          drift_valu, self.rotation[inx_t][0],
                                                          self.chamber_direction)
                    red_im = back_corrt(red_im.astype(np.float64))
                    red_channels[time] = [cropbox(red_im, cb) for cb in self.loaded_chamber_box]
                    red_time_points[time] = time_point
            return time

        _ = Parallel(n_jobs=128, require='sharedmem')(delayed(parallel_flur_seg)(inx_t, time)
                                                      for inx_t, time in enumerate(tqdm(self.times['phase'])))

        if 'green' in self.channels:
            self.time_points['green'] = [green_time_points[i] for i in self.times['green']]
        if 'red' in self.channels:
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
                    if 'green' in self.channels:
                        if time in self.times['green']:
                            green_channel_im = green_channels[time][cha_name_inx]
                            green_pixels = green_channel_im[cell_mask == 255]
                            # cell_pars['green_mean'] = cv2.mean(green_channel_im, cell_mask)[0]
                            cell_pars['green_mean'] = np.mean(green_pixels)
                            # medium only consider the medium of the brightest 10% pixels.
                            cell_pars['green_medium'] = np.quantile(green_pixels, 0.95)
                    if 'red' in self.channels:
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
        try:  # if have no mother
            self.dataframe_mother_cells = pd.concat(pf_list, sort=False)
            self.dataframe_mother_cells.index = pd.Index(range(len(self.dataframe_mother_cells)))
        except ValueError:
            print(f'[{self.fov_name}] -> Waring, has no mother cell.')
        return None

    def dump_data(self, compress=True):
        print(f"[{self.fov_name}] -> dump memory data.\n")
        if self.dataframe_mother_cells:
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
                         mother_cells_parameters=self.mother_cell_pars,
                         mother_cells_dataframe=self.dataframe_mother_cells
                         )
        if 'green' in self.channels:
            save_data['chamber_green_images'] = self.chamber_green_ims,
        if 'red' in self.channels:
            save_data['chamber_red_images'] = self.chamber_red_ims,
        if compress:
            dump(save_data, os.path.join(self.dir, self.fov_name + '.jl'), compress='lz4')
        print(f"[{self.fov_name}] -> memory data saved successfully.\n")
        return None

    def process_flow_GPU(self):
        print(f'[{self.fov_name}] -> detect channels.\n')
        self.detect_channels()
        print(f'[{self.fov_name}] -> detect frameshift.\n')
        self.detect_frameshift()
        print(f'[{self.fov_name}] -> detect cells.\n')
        self.cell_detection()

    def process_flow_CPU(self):
        print(f"[{self.fov_name}] -> extract cells' features.\n")
        self.extract_mother_cells_features()
        print(f"[{self.fov_name}] -> get mother cells data.\n")
        self.parse_mother_cell_data()
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
        fovs_name = [MomoFov(folder, DIR) for folder in fov_folder]
        return fovs_name


def image_conv(imas):
    chamber_im = imas
    len_im = chamber_im.shape[-2]
    v_means = []
    for i in range(len(chamber_im)):
        v_means.append(vertical_mean(chamber_im[i]))

    conv_sg = np.ones(len_im)
    for vm in v_means:
        vm = (vm - np.mean(vm)) / np.std(vm)
        conv_sg = conv_sg * fft(vm)

    freq = fftfreq(len_im, 1 / len_im)
    mask = freq > 0
    return conv_sg[mask], freq[mask]


# %%
if __name__ == '__main__':
    DIR = r'Z:\panchu\image\MoMa\20210101_NCM_pECJ3_M5_L3'

    untreated = get_fovs_name(DIR)

    for i, fov in enumerate(untreated):
        print(f'Processing {i + 1}/{len(untreated)}')
        fov.process_flow()
        del untreated[i]

    ######################################################
    # %%
    DIR = r'Z:\panchu\image\MoMa\20210101_NCM_pECJ3_M5_L3'
    # DIR = r'test_data_set/test_data'
    fovs_name = get_fovs_name(DIR, all_fov=True)
    fovs_num = len(fovs_name)

    chamber_info = []
    for fov_i in range(fovs_num):
        fov = fovs_name.pop()
        fov.detect_channels()
        load_index, grey, freq = fov.detect_frameshift()
        chambers_na = [f'{fov.fov_name}_{i}' for i in range(len(grey))]
        del fov
        load_list = [1 if ch_i in load_index else 0 for ch_i in range(len(grey))]
        load_pf = pd.DataFrame(data=dict(name=chambers_na, loaded=load_list, grey_value=grey, freq=freq))
        chamber_info.append(load_pf)
    load_df_all = pd.concat(chamber_info)
    load_df_all.index = pd.Index(np.arange(len(load_df_all)))
    load_df_all.to_csv(os.path.join(DIR, 'chamber_load', 'load_train_data.csv'))
#%%
    if_loaded = []
    sg_freq = []
    fig1, ax = plt.subplots(1, 1)
    for info in chamber_info:
        index = info[0]
        F = info[-1]
        for i in range(len(index)):
            sg, freq = F[i]
            sg_freq.append(F[i])
            if i in index:
                ax.plot(freq, np.log(abs(sg)), '--k', alpha=0.5)
                if_loaded.append([np.float(1)])
            else:
                ax.plot(freq, np.log(abs(sg)), '-r')
                if_loaded.append([np.float(0)])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('$\ln(\|F\|)$')
    fig1.show()

    from sklearn.decomposition import PCA
    from sklearn import svm

    load_x = []

    for i, load in enumerate(if_loaded):
        sg, freq = sg_freq[i]  # only use freq less than 25
        sg = abs(np.log(sg[5:20]))
        load_x.append(load + list(sg))

    load_x = np.array(load_x)

    train_x = load_x[:int(len(load_x) * 0.8), :]
    test_x = load_x[int(len(load_x) * 0.8):, :]

    load_model_pca = PCA(n_components=6)
    load_class_scv = svm.SVC()

    load_model_pca.fit(train_x[:, 1:])

    model_pca_results = load_model_pca.transform(train_x[:, 1:])

    load_class_scv.fit(model_pca_results, train_x[:, 0])

    fig1, ax = plt.subplots(1, 1)

    ax.scatter(model_pca_results[:, 0], model_pca_results[:, 1], c=[colors_2[int(i)] for i in train_x[:, 0]])

    fig1.show()

    reduction = load_model_pca.transform(load_x[:, 1:])
    predict = load_class_scv.predict(reduction)
    print(load_x[:, 0])
    predict
    predict - load_x[:, 0]
