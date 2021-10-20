# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
import platform
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
from tqdm import tqdm
import _thread as thread
from typing import Tuple, Union, Dict, List, Optional, Any
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from warnings import warn
import subprocess as sbp
from scipy.io import savemat
# […]


# Own modules
from utils.delta.data import postprocess
from utils.delta.model import unet_chambers, unet_seg
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

model_file = r'./test_data_set/model/chambers_id_tessiechamp_old.hdf5'
seg_model_file = r'./test_data_set/model/unet_moma_seg_multisets.hdf5'
min_chamber_area = 1500
target_size = (512, 512)
input_size = target_size + (1,)
process_size = 200
print('[Momo] -> Loading models')
model_chambers = unet_chambers(input_size=input_size)
model_chambers.load_weights(model_file)
target_size_seg = (256, 32)
model_seg = unet_seg(input_size=target_size_seg + (1,))
model_seg.load_weights(seg_model_file)
lock = thread.allocate_lock()
# %%
colors_2 = ['#FFA2A8', '#95FF57']  # red, green
global plf
plf = platform.system()


def move_img_subfold(base_dir, fold_name='phase'):
    cmd = f"mkdir {os.path.join(base_dir, fold_name)}"
    file_surfix = ['tiff', 'tif']
    if plf == 'Linux':
        OSTAG = True
    else:
        OSTAG = False
    sbp.run(cmd, shell=OSTAG)
    with os.scandir(base_dir) as file_it:
        files = [file.name for file in file_it if file.is_file()]
    files = [os.path.join(base_dir, file) for file in files if file.split('.')[-1] in file_surfix]
    cmd2 = f"mv -t {os.path.join(base_dir, fold_name)} {' '.join(files)}"
    sbp.run(cmd2, shell=OSTAG)
    return None


def get_channel_name(dir, name) -> list:
    with os.scandir(os.path.join(dir, name)) as file_it:
        dirs = [file.name for file in file_it if file.is_dir()]
    return dirs


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


def get_times(fov_dir_base: str, name: str, channels: List[str]) -> Dict[str, List[str]]:
    """

    :param fov_dir_base: str. fovs dir name
    :param name: str. fov dir name
    :param channels: list. channels name
    :return: dict: {channel_name:[list of file name]} int represents the acquisition order.
    """
    tim_dict = dict()
    img_suffix = ['tif', 'tiff']
    for channel in channels:
        file_names = os.listdir(os.path.join(fov_dir_base, name, channel))  # type: List[str]
        file_name = [fil_name for fil_name in file_names if fil_name.split(".")[-1] in img_suffix]
        file_name.sort(key=lambda elem: int(elem.split('.')[0].split('t')[-1]))  # example tiff file name: t20.tiff
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
    """
    get the image acq_tim

    Parameters
    ----------
    ps : str
        image path

    Returns
    ----------
    Tuple[np.ndarray, Optional[str]]
        tuple, including the image array and acquisition time.

    """
    with tif.TiffFile(ps) as tim:
        try:
            acq_tim = tim.shaped_metadata[0]['time']
        except TypeError:
            acq_tim = None

        return tim.asarray(), acq_tim


def get_fluo_channel(ps, drift, angle, direct):
    """


    """
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
    _ = Parallel(n_jobs=32, require='sharedmem')(delayed(resize_map)(im_inx, size) for im_inx in range(ims_num))
    return resize_ims, subims


def get_fovs(dir, all_fov=False, sort=True, **kwargs):
    """
    Get fovs name under dir, if fovs in dir have been treated (i.e. having a memory obj in folder dir), these fovs will
    not returned.
    :param dir: str, ps
    :param all_fov: bool, if True, return all folders, default, False.
    :return: list
    """
    fov_tag = ['fov', 'pos']
    DIR = dir
    jl_file = [jl_name.split('.')[0] for jl_name in os.listdir(DIR) if jl_name.split('.')[-1] == 'jl']
    fov_folder = [folder for folder in os.listdir(DIR)
                  if (folder[:3] in fov_tag and os.path.isdir(os.path.join(DIR, folder)))]

    if all_fov == False:
        untreated = list(set(fov_folder) - set(jl_file))
        if sort == True:
            untreated.sort(key=lambda name: int(name[3:].strip("_")))
        fovs_name = [MomoFov(folder, DIR, **kwargs) for folder in untreated]
        return fovs_name
    else:
        if sort == True:
            fov_folder.sort(key=lambda name: int(name[3:].strip("_")))
        fovs_name = [MomoFov(folder, DIR, **kwargs) for folder in fov_folder]
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


def cv_otsu(images: np.ndarray, gaussian_core=(3, 3)) -> Tuple[np.ndarray, list]:
    single_image_flag = False
    ims = images.copy()
    if len(ims.shape) == 2:
        ims = np.expand_dims(ims, axis=0)
        single_image_flag = True
    ims_num = ims.shape[0]
    thresholds = []
    for i in range(ims_num):
        im = ims[i, ...]
        im = rangescale(im, (0, 255)).astype(np.uint8)
        im = cv2.GaussianBlur(im, gaussian_core, 0)
        thre, ims[i, ...] = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.append(thre)
    if single_image_flag:
        ims = np.squeeze(ims)
        thresholds = thresholds[0]
    return ims, thresholds


def cv_full_contour2mask(cnt, size) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(size, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    points_index = np.transpose(np.nonzero(mask))
    return mask, points_index


def cv_edge_counter2mask(cnt, size, thickness=1) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(size, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, thickness)
    points_index = np.transpose(np.nonzero(mask))
    return mask, points_index


def cv_edge_fullmask2contour(img, thickness=2) -> Tuple[np.ndarray, np.ndarray]:
    """
    input a filled mask of cell, return the edge around the cell mask and the indexes of edge.
    :param img: binary ndarray, np.uint8
    :param thickness: the thickness of the edge, default is 2
    :return: (mask of edge, edge indexes)
    """
    shape = img.shape
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img, contour = cv_edge_counter2mask(contours[0], shape, thickness)
    return img, contour


def cv_out_edge_contour(img, kernel_size=3, axis=None, thickness=2):
    if axis == 0:
        kernel = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel[:, int(kernel_size // 2)] = np.ones(kernel_size, np.uint8)
    elif axis == 1:
        kernel = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel[int(kernel_size // 2), :] = np.ones(kernel_size, np.uint8)
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    diliation = cv2.dilate(img, kernel, iterations=1)
    diliation_img, countour_index = cv_edge_fullmask2contour(diliation, thickness)
    return diliation_img, countour_index


def cv_open(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.int8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def line_length_filter(img, length, axis):
    img = img.astype(np.float32)
    if axis is not None:
        ker_length = length * 2 - 1
        kernel = np.zeros((ker_length, ker_length), np.float32)
        if axis == 0:
            kernel[:, length - 1] = 1
        elif axis == 1:
            kernel[length - 1, :] = 1
        kernel[length - 1, length - 1] = - (length - 2)
    else:
        kernel = np.ones((3, 3), np.float32)
        kernel[1, 1] = -1
    dst = cv2.filter2D(img, -1, kernel)
    dst = dst > 0
    dst = np.logical_and(img, dst).astype(np.uint8) * 255
    return dst


class Cell:
    def __init__(self, cell_index, channels=None, umppx=0.065):
        if channels is None:
            channels = []
        self.cell_index = cell_index  # type: list  # [chamber_key, time_index, cell_index]
        self.channels = channels
        self.mask_init = None
        self.mask_xy = None
        self.contour_init = None
        self._mask_opt = None
        self._contour_opt = None
        self.skeleton = None
        self.edge_mask = None
        self.edge_mask_xy = None
        self.skeleton_func = None
        self.spine = None
        self.spine_length = None
        self.area = None
        self.umppx = umppx  # type: float  # nu m per pixel
        self.channel_imgs = None  # type: Optional[Dict[str, np.ndarray]]
        self.flu_level = None  # type: Optional[Dict[str, Any]]
        self.rectangle = None  # type: Optional[dict]

    def set_mask_init(self, cnt, size):
        self.contour_init = cnt
        self.mask_init, self.mask_xy = cv_full_contour2mask(cnt, size)
        self._mask_opt = self.mask_init.copy()
        self._contour_opt = self.contour_init.copy()

    @property
    def mask_opt(self):
        return self._mask_opt

    @mask_opt.setter
    def mask_opt(self, img):
        self._mask_opt = img
        contour_opt, _ = cv2.findContours(self._mask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._contour_opt = contour_opt[0]

    @property
    def mask(self):
        return self._mask_opt

    @property
    def contour(self):
        return self._contour_opt

    def cal_cell_skeleton(self, update=True):
        if update:
            self.edge_mask, self.edge_mask_xy = cv_edge_fullmask2contour(self._mask_opt)
        bins = int(np.ptp(self.edge_mask_xy[:, 0]) / 3)
        if bins < 5:
            bins = 5
        y_stat = binned_statistic(self.edge_mask_xy[:, 0], self.edge_mask_xy[:, 1],
                                  'mean', bins=bins)
        x_stat = np.diff(y_stat[-2]) / 2 + y_stat[-2][:-1]
        self.skeleton = np.hstack((x_stat.reshape(-1, 1), y_stat[0].reshape(-1, 1)))
        self.skeleton_func = UnivariateSpline(self.skeleton[:, 0], self.skeleton[:, 1], k=3, s=None)
        x_min, x_max = self._contour_opt[:, 0, 1].min(), self._contour_opt[:, 0, 1].max()
        x_space = np.linspace(x_min, x_max, endpoint=True)
        self.spine = np.hstack((x_space.reshape(-1, 1),
                                self.skeleton_func(x_space).reshape(-1, 1)))
        self.spine_length = np.sum(np.sqrt((self.skeleton_func.derivative()(x_space)) ** 2 + 1.)) \
                            * (x_space[1] - x_space[0]) * self.umppx
        rect = cv2.minAreaRect(self._contour_opt)
        self.rectangle = dict(
            rec_length=max(rect[1]),
            rec_width=min(rect[1])
        )
        self.area = cv2.contourArea(self._contour_opt) * self.umppx ** 2

    def assign_channel_img(self, channel_name: str, channel_img: np.ndarray):
        if channel_name in self.channels:
            warn(f"{self.cell_index[0]} {self.cell_index[1]} {self.cell_index[2]}: {channel_name} has "
                 f"already assigned, the new imported image will cover the old one.")
        self.channel_imgs[channel_name] = channel_img

    def assign_channel_imgs(self, channel_imgs: Dict[str, np.ndarray]):
        self.channel_imgs = channel_imgs
        self.channels = list(channel_imgs.keys())

    def cal_cel_flu_leve(self, channels: Union[str, list]):
        if isinstance(channels, str):
            channels = [channels]
        channels = [channel for channel in channels if channel in self.channels]
        if channels:
            self.flu_level = {}
            for channel in channels:
                flu_img = self.channel_imgs[channel]
                # print(self._mask_opt)
                flu_pixels = flu_img[self._mask_opt == 255]
                flu_avg = np.mean(flu_pixels)
                flu_medium = np.quantile(flu_pixels, 0.95)
                self.flu_level[channel] = dict(mean=flu_avg, medium=flu_medium)


class MomoFov:
    loaded_chamber_name: List[str]  # chamber names of loaded
    times: Union[Dict[str, List[str]], None] = None

    def __init__(self, name, fov_dir_base: str, exp_mode=False,
                 cell_minial_px: int = 100, time_step=180.,
                 cell_detection='phase', quantify=None):
        """
        :param exp_mode:
        :param cell_detection:
        :param quantify:
        :param name: str, fov name eg. 'fov_0'
        :param fov_dir_base: str, path of fovs
        """
        if quantify is None:
            quantify = ['red', 'green']
        self.fov_name = name  # type: str
        self.dir = fov_dir_base  # type: str  # where the fov fold contain.
        self.cell_mini_size = cell_minial_px  # type: int
        self.channels = get_channel_name(self.dir, self.fov_name)  # type: List[str] # channel names in list
        if len(self.channels) == 0:
            self.channels = [cell_detection]
            move_img_subfold(os.path.join(self.dir, self.fov_name), fold_name='phase')
        quantify = [ch for ch in quantify if ch in self.channels]
        self.channels_function = {"cell_detection": cell_detection, "quantify": quantify}
        self.times = None
        self.phase_ims = None  # type: Union[np.ndarray, None]  # [time number, 0 axis, 1 axis]
        self.time_points = dict()  # type: dict[str, dict[str, str]]
        self.chamber_mask = None
        self.chamber_boxes = []
        self.loaded_chamber_box = []  # type: list
        self.drift_template = None
        self.template_frame = None
        self.image_size = None
        # if chamber outlet towards to top in fov, the value is 0.
        self.chamber_direction = None  # type:Union[int, None]
        self.drift_values = None
        # NOTE: cell_mask have a order which is dependent on times. i.e. #chan_1 ---times----
        # #chan_2 --- times---
        self.cell_mask = None
        self.chamber_cells_mask = dict()  # type: Dict[str, np.ndarray]  # {chamber_name:[time, axis0, axis1]}
        # {chamber_name: list of time[list of cells[ndarray contours along axis0], ...]}
        self.chamber_cells_contour = dict()  # type: Dict[str, List[List[np.ndarray]]]
        self.loaded_chamber_name = []  # type: List[str]
        self.mother_cell_pars = dict()
        self.dataframe_mother_cells = None
        self.index_of_loaded_chamber = []  # type: List
        self.chamber_gray_level = []
        self.chamber_seg = None
        self.rotation = []
        self.experiment_mode = exp_mode  # type: bool
        self.ims_channels_dict = {ch: f"chamber_{ch}_ims" for ch in self.channels}
        self.time_step = time_step  # type: float
        # cells is a dict. dict[chamber_name: dict[time: list[Cell]]]
        self.cells = None  # type: Optional[Dict[str, Dict[str,List[Cell]]]]
        self.fields_init()

    def fields_init(self):
        for ch in self.channels:
            self.__dict__[self.ims_channels_dict[ch]] = dict()  # type: Dict[str, Dict[str, np.ndarray]]

    def fmt_str(self, msg):
        return f"[{self.fov_name}] -> {msg}"

    def detect_channels(self, index=0):
        """
        The frame index used to detect side_channels. default is 0
        images inputted and rotated than vertical flip if needed, than detect the fame shift
        :param index: int,
        :return:
        """
        channel_detect_cell = self.channels_function['cell_detection']
        self.times = get_times(self.dir, self.fov_name, self.channels)
        im, _ = get_im_time(os.path.join(self.dir, self.fov_name, channel_detect_cell,
                                         self.times[channel_detect_cell][index]))
        self.image_size = im.shape
        im, _ = rotate_fov(np.expand_dims(im, axis=0), crop=False)
        im = rangescale(im.squeeze(), rescale=(0, 1.))
        self.template_frame = im
        # TODO: if the images are not 2048 x 2048.
        back_ground = np.ones((2048, 2048)) * np.median(im)
        y_bl, x_bl = tuple([round((2048 - length) / 2) for length in im.shape])
        back_ground[y_bl:(y_bl + im.shape[0]), x_bl:(x_bl + im.shape[1])] = im.copy()
        first_frame = np.expand_dims(np.expand_dims(cv2.resize(back_ground.squeeze(), (512, 512)), axis=0), axis=3)
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(first_frame.squeeze())
        # fig.show()
        # using expand_dims to get it into a shape that the chambers id unet accepts
        # Find chambers, filter results, get bounding boxes:
        chamber_mask = model_chambers.predict(first_frame, verbose=0)
        chamber_mask = cv2.resize(np.squeeze(chamber_mask), back_ground.shape)
        chamber_mask = chamber_mask[y_bl:(y_bl + im.shape[0]), x_bl:(x_bl + im.shape[1])]
        # scaling back to original size
        # chamber_mask = rangescale(chamber_mask, (0, 255)).astype(np.uint8)
        # chamber_mask, _ = cv_otsu(chamber_mask, gaussian_core=(9, 9))
        chamber_mask = postprocess(chamber_mask, min_size=min_chamber_area)  # Binarization, cleaning
        # and area filtering
        self.chamber_boxes = getChamberBoxes(np.squeeze(chamber_mask))

        print(f"[{self.fov_name}] -> detect {len(self.chamber_boxes)} chambers.")
        border = int(im.shape[0] * 0.02)
        chambers_border = dict(xtl=min(self.chamber_boxes, key=lambda elem: elem['xtl'])['xtl'],
                               xbr=max(self.chamber_boxes, key=lambda elem: elem['xbr'])['xbr'],
                               ytl=min(self.chamber_boxes, key=lambda elem: elem['ytl'])['ytl'] - border,
                               ybr=max(self.chamber_boxes, key=lambda elem: elem['ybr'])['ybr'] + border
                               )  # get the borders of all channels.

        # make sure the border do not extend the image frame
        chambermask_ylength, _ = chamber_mask.shape
        if chambers_border['ytl'] < 0:
            chambers_border['ytl'] = 0
        if chambers_border['ybr'] > chambermask_ylength:
            chambers_border['ybr'] = chambermask_ylength

        v_m = vertical_mean(cropbox(self.template_frame, chambers_border))
        # image vertical flip if chamber outlet in the top of the frame.
        if np.mean(v_m[0:(len(v_m) // 2)]) >= np.mean(v_m[(len(v_m) // 2):]):
            self.chamber_direction = 0
            self.chamber_mask = chamber_mask[::-1, :]
            self.chamber_boxes = getChamberBoxes(self.chamber_mask)
            self.drift_template = getDriftTemplate(self.chamber_boxes, im.squeeze()[::-1, :])
        else:
            self.chamber_direction = 1
            self.chamber_mask = chamber_mask
            self.drift_template = getDriftTemplate(self.chamber_boxes, im.squeeze())

        return None

    def detect_frameshift(self):
        """
        This function used to detect frame shift.
        :return: None
        """
        channel_detect_cell = self.channels_function['cell_detection']
        print(f'[{self.fov_name}] -> loading phase images.')
        self.phase_ims = np.zeros((len(self.times[channel_detect_cell]),) + self.template_frame.shape)
        self.time_points[channel_detect_cell] = {}
        self.rotation = [None] * len(self.times[channel_detect_cell])

        def parallel_input(fn, inx):
            """
            fn: file name (tXX.tiff)
            index: index of time
            """
            lock.acquire()  # async io is important when using linux platform.
            im, tp = get_im_time(os.path.join(self.dir, self.fov_name, channel_detect_cell, fn))
            self.time_points[channel_detect_cell][fn] = tp
            lock.release()
            im, angl = rotate_fov(np.expand_dims(im, axis=0), crop=False)
            if self.chamber_direction == 0:
                im = im[:, ::-1, :]
            self.phase_ims[inx, ...] = rangescale(im.squeeze(), (0, 1))

            self.rotation[inx] = angl
            return None

        def parallel_input_processing(fn, dirct, dir, fov_name):
            im, tp = get_im_time(os.path.join(dir, fov_name, channel_detect_cell, fn))
            im, angl = rotate_fov(np.expand_dims(im, axis=0), crop=False)
            if dirct == 0:
                im = im[:, ::-1, :]
            return dict(im=rangescale(im.squeeze(), (0, 1)), tp=tp, angl=angl)

        # --------------------- input all phase images --------------------------------------
        _ = Parallel(n_jobs=64, require='sharedmem')(
            delayed(parallel_input)(fn, i)
            for i, fn in enumerate(tqdm(self.times[channel_detect_cell], desc=self.fmt_str("loading phase images"))))
        if None in self.time_points[channel_detect_cell].values():
            times = np.arange(len(self.time_points[channel_detect_cell].values())) * self.time_step
            for index, time_key in enumerate(self.time_points[channel_detect_cell].keys()):
                self.time_points[channel_detect_cell][time_key] = times[index]

        # if plf != 'Linux':
        #     _ = Parallel(n_jobs=64, require='sharedmem')(
        #         delayed(parallel_input)(fn, i) for i, fn in enumerate(tqdm(self.times['phase'])))
        # else:
        #     print(f'[{self.fov_name}] -> loading phase images by multi-processing.')
        #     input_im_meta = Parallel(n_jobs=1)(
        #         delayed(parallel_input_processing)(fn, self.chamber_direction, self.dir, self.fov_name)
        #         for fn in tqdm(self.times['phase']))
        #     for time_i in range(len(input_im_meta)):
        #         self.phase_ims[time_i, ...] = input_im_meta[time_i]['im']
        #         self.time_points['phase'][time_i] = input_im_meta[time_i]['tp']
        #         self.rotation[time_i] = input_im_meta[time_i]['angl']

        print(f'[{self.fov_name}] -> ims shape is {self.phase_ims.shape}.')
        # --------------------- input all phase images --------------------------------------
        drift_cor_box = dict(xtl=0,
                             xbr=None,
                             ytl=0,
                             ybr=max(self.chamber_boxes, key=lambda elem: elem['ytl'])['ytl']
                             )  # Box to match template
        self.phase_ims, self.drift_values = driftcorr(self.phase_ims,
                                                      template=self.drift_template, box=drift_cor_box)
        xcorr_one = int(self.drift_values[0][0])
        ycorr_one = int(self.drift_values[1][0])
        for box in self.chamber_boxes:  # TODO: frame shift have bug.
            box['xtl'] -= xcorr_one
            box['xbr'] -= xcorr_one
            box['ytl'] -= ycorr_one
            box['ybr'] -= ycorr_one
        # -------------detect whether chambers were loaded with cells--------
        num_time = len(self.times[channel_detect_cell])
        sample_index = np.random.choice(range(int(num_time * 0.1)), 10)
        selected_ims = self.phase_ims[sample_index, ...]

        chamber_graylevel = []
        chamber_frq = []

        for box in self.chamber_boxes:
            half_chambers = selected_ims[:,
                            box['ytl']:int((box['ybr'] - box['ytl']) / 2 + box['ytl']),
                            box['xtl']:box['xbr']]
            mean_chamber = np.mean(half_chambers)
            chamber_graylevel.append(mean_chamber)
            if self.experiment_mode:
                sg, frq = image_conv(np.expand_dims(selected_ims[0,
                                                    int(box['ytl'] + (box['ybr'] - box['ytl']) * 0.20):int(
                                                        (box['ybr'] - box['ytl']) * 0.6 + box['ytl']),
                                                    box['xtl']:box['xbr']], axis=0))
                chamber_frq.append([sg, frq])

        # cells_threshold = np.min(chamber_graylevel) + np.ptp(chamber_graylevel) * 0.8
        cells_threshold = np.min(chamber_graylevel) + np.ptp(chamber_graylevel) * 0.8
        chamber_loaded = [True if value < cells_threshold else False for value in chamber_graylevel]
        self.index_of_loaded_chamber = list(np.where(chamber_loaded)[0])
        self.loaded_chamber_box = [self.chamber_boxes[index] for index in self.index_of_loaded_chamber]

        if self.experiment_mode:
            fig_sg, ax = plt.subplots(1, 1)
            for i in range(len(chamber_frq)):
                sg, frq = chamber_frq[i]
                if i in self.index_of_loaded_chamber:
                    ax.plot(frq, np.log(abs(sg)), '--k', alpha=0.2)
                else:
                    ax.plot(frq, np.log(abs(sg)), '-r')

            fig_sg_ps = os.path.join(self.dir, 'chamber_load')
            try:
                os.makedirs(fig_sg_ps)
            except FileExistsError:
                print(f"[{self.fov_name}] -> {fig_sg_ps} is existed!")
            plt.savefig(os.path.join(fig_sg_ps, f'{self.fov_name}_F.svg'), format='svg')

            fig_ch, ax = plt.subplots(1, 1)
            colors = [colors_2[1] if i in self.index_of_loaded_chamber else colors_2[0]
                      for i in range(len(chamber_frq))]
            draw_channel_order(rangescale(self.phase_ims[0, ...], (0, 255)).astype(np.uint8),
                               self.chamber_boxes, colors, ax=ax)
            plt.savefig(os.path.join(fig_sg_ps, f'{self.fov_name}_chamber.svg'), format='svg')

        self.chamber_gray_level = chamber_graylevel
        # print(f'[{self.fov_name}] -> , chamber_graylevel)
        print(f'[{self.fov_name}] -> detect loaded chamber number: {len(self.loaded_chamber_box)}.')
        return [self.index_of_loaded_chamber, self.chamber_gray_level, chamber_frq]

    def cell_detection(self):
        channel_detect_cell = self.channels_function['cell_detection']
        chmber_ims = self.__dict__[self.ims_channels_dict[channel_detect_cell]]
        seg_inputs = ()
        for m, chamberbox in enumerate(self.loaded_chamber_box):
            if chamberbox:
                sub_inputs, ori_cham_imgs = parallel_seg_input(self.phase_ims, chamberbox)
                seg_inputs += (sub_inputs,)
                ori_imgs_dict = {t: ori_cham_imgs[i, ...]
                                 for i, t in enumerate(self.times[channel_detect_cell])}
                chmber_ims[f'ch_{self.index_of_loaded_chamber[m]}'] = ori_imgs_dict
        seg_inputs = np.copy(np.concatenate(seg_inputs, axis=0))
        del self.phase_ims  # release memory
        seg_inputs = np.expand_dims(np.array(seg_inputs), axis=3)
        self.chamber_seg = seg_inputs
        # Format into 4D tensor
        # Run segmentation U-Net:
        seg = model_seg.predict(seg_inputs)
        self.cell_mask = postprocess(seg[:, :, :, 0], min_size=self.cell_mini_size, square_size=5)

        # -------------- reform the size--------------
        def parallel_rearange_mask(t, chn):
            frame_index = t + chn * len(self.times[channel_detect_cell])
            ori_frames[t] = cv2.resize(self.cell_mask[frame_index], ori_frames.shape[2:0:-1])

        for m, box in enumerate(self.loaded_chamber_box):
            ori_frames = np.empty([len(self.times[channel_detect_cell]),
                                   box['ybr'] - box['ytl'], box['xbr'] - box['xtl']]).astype(np.uint16)

            # rerange_mask = [dask.delayed(parallel_rearange_mask)(t, m) for t in range(len(self.times['phase']))]
            # _ = dask.compute(*rerange_mask)

            _ = Parallel(n_jobs=64, require='sharedmem')(delayed(parallel_rearange_mask)(t, m)
                                                         for t in range(len(self.times[channel_detect_cell])))

            self.chamber_cells_mask[f'ch_{self.index_of_loaded_chamber[m]}'] = ori_frames
        # -------------- get cells contour ------------
        self.loaded_chamber_name = list(self.chamber_cells_mask.keys())
        for chamber in self.loaded_chamber_name:
            contours_list = []
            for time_mask in self.chamber_cells_mask[chamber]:
                contours = cv2.findContours((time_mask > 0).astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0]  # Get contours of single cells
                contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sort along 0 axis
                contours_list.append(contours)
            self.chamber_cells_contour[chamber] = contours_list

    def extract_cells_features(self, maxium_iter=3):
        # TODO: bug: when channel have no cells, the features results are strange.

        for channel in self.channels_function['quantify']:
            self.__dict__[self.ims_channels_dict[channel]] = {chamber_na: {}
                                                              for chamber_na in self.loaded_chamber_name}
            self.time_points[channel] = {}

        def parallel_flur_seg(inx_t, time):
            """
            get all fluorescent images from disk and seg into channel XX_channels is a dictionary keys are file 
            names and their elements are lists containing chambers ordered by loaded chamber in 
            self.loaded_chamber_name. 
            """
            for channel in self.channels_function['quantify']:
                ims_channels_dict = self.__dict__[self.ims_channels_dict[channel]]
                if time in self.times[channel]:
                    drift_valu = (self.drift_values[0][inx_t], self.drift_values[1][inx_t])
                    lock.acquire()
                    # read and reform the flu-image
                    flu_ims, time_point = get_fluo_channel(os.path.join(self.dir, self.fov_name, channel, time),
                                                           drift_valu, self.rotation[inx_t][0],
                                                           self.chamber_direction)
                    lock.release()
                    flu_ims = back_corrt(flu_ims.astype(np.float64))  # fluorescence background correct

                    for i, cb_name in enumerate(self.loaded_chamber_name):
                        ims_channels_dict[cb_name][time] = cropbox(flu_ims, self.loaded_chamber_box[i])

                    self.time_points[channel][time] = time_point
            return None

        pb_msg = self.fmt_str("loading fluorescent images")
        if self.channels_function['quantify']:
            if plf != 'Linux':
                _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parallel_flur_seg)(inx_t, time)
                                                             for inx_t, time in enumerate(tqdm(self.times['phase'],
                                                                                               desc=pb_msg)))
            else:
                _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parallel_flur_seg)(inx_t, time)
                                                             for inx_t, time in enumerate(tqdm(self.times['phase'],
                                                                                               desc=pb_msg)))

        self.cells = {chamber_name: {} for chamber_name in self.loaded_chamber_name}
        print(f'[{self.fov_name}] -> Optimize cell contour.')
        for cha_name_inx, chamber_name in enumerate(tqdm(self.loaded_chamber_name,
                                                         desc=self.fmt_str("Optimize cell contour"))):
            detect_channel_key = self.channels_function["cell_detection"]
            detect_imgs_dict = self.__dict__[self.ims_channels_dict[detect_channel_key]]
            for tm_inx, time in enumerate(self.times[detect_channel_key]):
                cells_contour = self.chamber_cells_contour[chamber_name][tm_inx]
                cells_number = len(self.chamber_cells_contour[chamber_name][tm_inx])
                if cells_contour:
                    phase_image = detect_imgs_dict[chamber_name][time]
                    chamber_mask = np.ones(phase_image.shape, np.uint8)
                    min_y_border = np.min([cnt[:, 0, 1].min() for cnt in cells_contour])
                    lim_y_border = min_y_border - 2
                    if lim_y_border < 1:
                        lim_y_border = min_y_border
                    chamber_mask[:lim_y_border, :] = 0
                    invert_phase = 1. - phase_image
                    # image_y_len = phase_image.shape[1]
                    # border = int(image_y_len / (2 * 0.02 + 1))
                    cells_list = [Cell(cell_index=[chamber_name, time, i]) for i in range(cells_number)]
                    self.cells[chamber_name][time] = cells_list
                    channel_imgs = {}
                    for channel in self.channels:
                        if time in self.times[channel]:
                            channel_imgs[channel] = self.__dict__[self.ims_channels_dict[channel]][chamber_name][time]
                    for index, cell in enumerate(cells_list):
                        cell.set_mask_init(cells_contour[index], phase_image.shape)
                        cell.assign_channel_imgs(channel_imgs)
                    # cell_mask_ori = cell_mask.copy()
                    iter_key = [True] * cells_number
                    for i in range(maxium_iter):
                        for index, cell in enumerate(cells_list):
                            if iter_key[index]:
                                cell_mask_od = cell.mask.copy()
                                cell_mask = cell.mask.copy()
                                cell_mask_edge, mask_edge_xy = cv_edge_fullmask2contour(cell_mask, thickness=2)
                                cell_mask_edge = line_length_filter(cell_mask_edge, 6, 0)
                                cell_mask_out, mask_out_xy = cv_out_edge_contour(cell_mask, axis=0)
                                cell_mask_xaxis = np.logical_and(cell_mask_out, np.logical_not(cell_mask_edge))
                                edge_ceiling_threshold = np.quantile(invert_phase[cell_mask != 0], 0.5)
                                masked_phase = invert_phase.copy()
                                masked_phase[cell_mask == 0] = 0
                                _, edge_floor_threshold = cv_otsu(masked_phase)
                                edge_floor_threshold /= 255
                                edge_revise_mask = np.logical_and((invert_phase < edge_floor_threshold), cell_mask_edge)
                                out_revise_mask = np.logical_and((invert_phase > edge_ceiling_threshold),
                                                                 cell_mask_xaxis)
                                cell_mask[edge_revise_mask] = 0
                                cell_mask[out_revise_mask] = 255
                                cell_mask = cv_open(cell_mask)
                                other_cells = cells_list[:index] + cells_list[index + 1:]
                                other_cells_mask = [c.mask_opt for c in other_cells]
                                other_cells_mask = np.logical_not(np.sum(other_cells_mask, axis=0))
                                cell_mask = np.logical_and(cell_mask, other_cells_mask)
                                cell_mask = np.logical_and(cell_mask, chamber_mask).astype(np.uint8)
                                if cell_mask.any():
                                    cell.mask_opt = cell_mask * 255
                                    check_array = cell.mask_opt == cell_mask_od
                                    if check_array.all():
                                        iter_key[index] = False
                                else:
                                    iter_key[index] = False
                        if True not in iter_key:
                            break
                    for cell in cells_list:
                        cell.cal_cell_skeleton()
                        cell.cal_cel_flu_leve(self.channels_function["quantify"])

    def parse_mother_cell_data(self):
        """
        parse all statistics data into one frame.
        :return:
        """
        pf_list = []
        for chamber in self.loaded_chamber_name:
            self.mother_cell_pars[chamber] = []
            for time_key, cells in self.cells[chamber].items():
                mother_cell = cells[0]  # mother cell
                time_point = self.time_points[self.channels_function["cell_detection"]][time_key]
                mother_cell_dict = dict(spine_length=mother_cell.spine_length,
                                        area=mother_cell.area,
                                        time_point=time_point)
                mother_cell_dict.update(mother_cell.rectangle)
                if mother_cell.flu_level is not None:
                    flu_channel_keys = list(mother_cell.flu_level.keys())
                    flu_channels_dict = dict()
                    for key in flu_channel_keys:
                        flu_dict = mother_cell.flu_level[key]
                        for stat_key, stat_value in list(flu_dict.items()):
                            flu_channels_dict[f"{key}_{stat_key}"] = stat_value
                    mother_cell_dict.update(flu_channels_dict)
                self.mother_cell_pars[chamber].append(mother_cell_dict)
            if self.mother_cell_pars[chamber]:
                pf = pd.DataFrame(data=self.mother_cell_pars[chamber])
                try:
                    time = [float(t.split(',')[-1]) for t in pf['time_point']]
                except AttributeError:
                    time = [float(t) for t in pf['time_point']]
                pf['time_s'] = time
                pf['chamber'] = [f'{self.fov_name}_{chamber}'] * len(pf)
                pf_list.append(pf)
        try:  # if have no mother
            self.dataframe_mother_cells = pd.concat(pf_list, sort=False)
            self.dataframe_mother_cells.index = pd.Index(range(len(self.dataframe_mother_cells)))
        except ValueError:
            print(f'[{self.fov_name}] -> Waring, has no mother cell.')
        return None

    def relink_cells_musk(self, mat_mask=False):
        detect_channel_key = self.channels_function["cell_detection"]
        fov_cells_mask_tensor = np.zeros((len(self.times[detect_channel_key]),) + self.image_size)
        for tm_inx, time in enumerate(self.times[detect_channel_key]):
            mask_template = np.zeros(self.image_size)
            for chamber_index, chamber_name in enumerate(self.loaded_chamber_name):
                box = self.loaded_chamber_box[chamber_index]
                try:
                    chamber_cells = self.cells[chamber_name][time]
                    chamber_cells_musk = np.sum([(cell_index + 1) * cell.mask_opt.astype(bool)
                                                 for cell_index, cell in enumerate(chamber_cells)], axis=0)
                    mask_template[box['ytl']:box['ybr'], box['xtl']:box['xbr']] = chamber_cells_musk
                except KeyError:  # no cell in chamber at that time
                    pass
            if self.chamber_direction == 0:
                mask_template = mask_template[::-1, ...]
            fov_cells_mask_tensor[tm_inx, ...] = mask_template  # (xcorr, ycorr)
        if mat_mask:
            dump_mat = dict(file_name=self.times[detect_channel_key],
                            rotation=self.rotation,
                            driftcorr=self.drift_values,
                            cells_mask=fov_cells_mask_tensor.astype(np.uint8))
            print(self.fmt_str('dumping mat file.'))
            savemat(os.path.join(self.dir, self.fov_name + '.mat'), dump_mat)

    def dump_data(self, compress=True):
        print(f"[{self.fov_name}] -> dump memory data.")
        if isinstance(self.dataframe_mother_cells, pd.DataFrame):
            self.dataframe_mother_cells.to_csv(os.path.join(self.dir, self.fov_name + '_statistic.csv'))
        save_data = dict(directory=self.dir,
                         fov_name=self.fov_name,
                         frame_rotation_anle=self.rotation,
                         frame_shift=self.drift_values,
                         times=self.times,
                         time_points=self.time_points,
                         light_channels=self.channels,
                         channels_im_dic_name=self.ims_channels_dict,
                         chamber_box=self.chamber_boxes,
                         chamber_loaded_name=self.loaded_chamber_name,
                         chamber_loaded_index=self.index_of_loaded_chamber,
                         chamber_grayvalue=self.chamber_gray_level,
                         chamber_direction=self.chamber_direction,
                         chamber_cells_mask=self.chamber_cells_mask,
                         chamber_cells_contour=self.chamber_cells_contour,
                         mother_cells_parameters=self.mother_cell_pars,
                         mother_cells_dataframe=self.dataframe_mother_cells,
                         cells_obj=self.cells
                         )
        for ch in self.channels:
            save_data.update({self.ims_channels_dict[ch]:
                                  self.__dict__[self.ims_channels_dict[ch]]})

        if compress:
            dump(save_data, os.path.join(self.dir, self.fov_name + '.jl'), compress='lz4')
        print(f"[{self.fov_name}] -> memory data saved successfully.")
        return None

    def process_flow_GPU(self):
        print(f'[{self.fov_name}] -> detect channels.')
        self.detect_channels()
        print(f'[{self.fov_name}] -> detect frameshift.')
        self.detect_frameshift()
        print(f'[{self.fov_name}] -> detect cells.')
        self.cell_detection()

    def process_flow_CPU(self):
        print(f"[{self.fov_name}] -> extract cells' features.")
        self.extract_cells_features()
        print(f"[{self.fov_name}] -> get mother cells data.")
        self.parse_mother_cell_data()
        self.dump_data()
        return None


# %%
if __name__ == '__main__':
    # %%
    # DIR = r'Z:\panchu\image\MoMa\20210101_NCM_pECJ3_M5_L3'
    # DIR = r'/data/20210225_pECJ3_M5_L3'
    # DIR = r"/media/fulab/4F02D2702FE474A3/MZX"
    # DIR = r"/home/fulab/data2/ZZ"
    DIR = r'test_data_set/test_data'

    fovs_name = get_fovs(DIR, time_step=120, all_fov=False)
    fovs_num = len(fovs_name)

    for fov in fovs_name:
        fov.process_flow_GPU()
        print(f"[{fov.fov_name}] -> extract cells' features.")
        fov.extract_cells_features()
        print(f"[{fov.fov_name}] -> get mother cells data.")
        fov.parse_mother_cell_data()
        fov.relink_cells_musk()
        fov.dump_data()
        # del fov
    # fov1 = fovs_name[0]
    #
    # fov1.detect_channels()
    # fov1.detect_frameshift()
    # fov1.cell_detection()
    # # %%
    # channel_index, time_index, cell_index = 0, time, 0
    # channel_key = fov1.loaded_chamber_name[1]
    # z_len, x_len, y_len = fov1.chamber_phase_ims[channel_key].shape
    # rand_choose = np.random.choice(range(z_len), 4, replace=False)
    # avg_phase = np.mean(fov1.chamber_phase_ims[channel_key][rand_choose, ...], axis=0)
    # image_x_profile = np.mean(avg_phase[int(x_len / 4):int(x_len * 3 / 4), :], axis=0)
    # image_x_profile = np.convolve(image_x_profile, np.ones(2) / 2, mode='same')
    # diff_image_x_profile = np.diff(image_x_profile)
    # sign_diff_x = np.sign(diff_image_x_profile)
    # diff_sign = np.diff(sign_diff_x)
    # borders = np.where(diff_sign == -2)[0]
    # border_left, border_right = borders.min() - 3, borders.max() + 3
    # chamber_mask = np.ones((x_len, y_len), np.uint8)
    # chamber_mask[:, 0:border_left] = 0
    # chamber_mask[:, border_right:] = 0
    #
    # for time in range(10):
    #
    #     cells_number = len(fov1.chamber_cells_contour[channel_key][time])
    #     cells_contour = fov1.chamber_cells_contour[channel_key][time]
    #     phase_image = fov1.chamber_phase_ims[channel_key][time, ...]
    #     min_y_border = np.min([cnt[:, 0, 1].min() for cnt in cells_contour])
    #     chamber_mask[:min_y_border - 2, :] = 0
    #     invert_phase = 1. - phase_image
    #     image_y_len = phase_image.shape[1]
    #     border = int(image_y_len / (2 * 0.02 + 1))
    #     cells_list = [Cell(cell_index=[channel_key, time, i]) for i in range(cells_number)]
    #
    #     for index, cell in enumerate(cells_list):
    #         cell.set_mask_init(cells_contour[index], (x_len, y_len))
    #
    #     # cell_mask_ori = cell_mask.copy()
    #     iter_key = [True] * cells_number
    #     for i in range(5):
    #         print(i)
    #         for index, cell in enumerate(cells_list):
    #             if iter_key[index]:
    #                 cell_mask_od = cell.mask.copy()
    #                 cell_mask = cell.mask
    #                 cell_mask_edge, mask_edge_xy = cv_edge_fullmask2contour(cell_mask)
    #                 cell_mask_edge = line_length_filter(cell_mask_edge, 6, 0)
    #                 cell_mask_out, mask_out_xy = cv_out_edge_contour(cell.mask, axis=0)
    #                 cell_mask_xaxis = np.logical_and(cell_mask_out, np.logical_not(cell_mask_edge))
    #                 edge_ceiling_threshold = np.quantile(invert_phase[cell.mask != 0], 0.45)
    #                 masked_phase = invert_phase.copy()
    #                 masked_phase[cell.mask_opt == 0] = 0
    #                 _, edge_floor_threshold = cv_otsu(masked_phase)
    #                 edge_floor_threshold /= 255
    #                 edge_revise_mask = np.logical_and((invert_phase < edge_floor_threshold), cell_mask_edge)
    #                 out_revise_mask = np.logical_and((invert_phase > edge_ceiling_threshold), cell_mask_xaxis)
    #                 cell_mask[edge_revise_mask] = 0
    #                 cell_mask[out_revise_mask] = 255
    #                 cell_mask = cv_open(cell_mask)
    #                 other_cells = cells_list[:index] + cells_list[index + 1:]
    #                 other_cells_mask = [c.mask_opt for c in other_cells]
    #                 other_cells_mask = np.logical_not(np.sum(other_cells_mask, axis=0))
    #                 cell_mask = np.logical_and(cell_mask, other_cells_mask)
    #                 cell_mask = np.logical_and(cell_mask, chamber_mask).astype(np.uint8)
    #                 cell.mask_opt = cell_mask
    #                 if (cell_mask == cell_mask_od).all():
    #                     iter_key[index] = False
    #                 print(iter_key)
    #         if True not in iter_key:
    #             break
    #
    #     masked_phase = 1. - phase_image.copy()
    #     masked_phase[cell_mask == 0] = 0
    #     otsu_image, otsu_thre = cv_otsu(masked_phase.copy())
    #
    #     invert_phase_rm_channel = 1 - phase_image.copy()
    #     invert_phase_rm_channel[chamber_mask == 0] = 0
    #     cell_pixels_medium = np.median(masked_phase[cell_mask != 0])
    #     cell_pixels_inerquantil = np.quantile(masked_phase[cell_mask != 0], 0.75) - \
    #                               np.quantile(masked_phase[cell_mask != 0], 0.25)
    #     tau_thrd = cell_pixels_medium - 1.5 * cell_pixels_inerquantil
    #     threshold_image = 1. - phase_image.copy()
    #     # threshold_image[cell_mask == 0] = 0
    #     threshold_image[threshold_image <= tau_thrd] = 0
    #
    #     all_mask_init = np.sum([c.mask_init for c in cells_list], axis=0)
    #     all_mask_opt = np.sum([c.mask_opt for c in cells_list], axis=0)
    #     all_skeleton = [c.cal_cell_skeleton() for c in cells_list]
    #     fig1, ax1 = plt.subplots(1, 10)
    #     ax1[0].imshow(all_mask_init)
    #     ax1[1].imshow(all_mask_opt)
    #     ax1[2].imshow(invert_phase_rm_channel, cmap='Greys')
    #     ax1[3].scatter(range(len(image_x_profile)), image_x_profile)
    #     # ax1[4].scatter(range(len(image_y_profile)), image_y_profile)
    #     ax1[5].imshow(line_length_filter(cell_mask_xaxis, 3, 1))
    #     ax1[6].hist(masked_phase[cell_mask != 0])
    #     ax1[7].imshow(cell_mask_edge)
    #     ax1[8].imshow(cell_mask_xaxis)
    #     # ax1[8].set_title(otsu_thre)
    #     # ax1[9].scatter(bin_avg[0], bin_x)
    #     ax1[9].imshow(1 - phase_image, cmap='Greys')
    #     for c in cells_list:
    #         ax1[9].scatter(c.skeleton[:, 1], c.skeleton[:, 0])
    #         ax1[9].plot(c.spine[:, 1], c.spine[:, 0], label="%.2f" % c.spine_length)
    #         ax1[9].legend()
    #     ax1[9].set_ylim(0, 328)
    #     ax1[9].set_xlim(0, 33)
    #     ax1[9].set_ylim(ax1[9].get_ylim()[::-1])  # invert the axis
    #     ax1[9].xaxis.tick_top()  # and move the X-Axis
    #     # ax1[9].yaxis.set_ticks(np.arange(0, 16, 1))  # set y-ticks
    #     ax1[9].yaxis.tick_left()  # remove right y-Ticks
    #     fig1.show()

    # DIR = r"/media/fulab/TOSHIBA_EXT/MZX"
    # fov_dir = os.listdir(DIR)
    # for fov in fov_dir:
    #     if 'phase' in os.listdir(os.path.join(DIR, fov)):
    #         cmd = f"rm {os.path.join(DIR, fov, 'phase')}"
    #
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(fov.chamber_mask)
    # ax[1].imshow(fov.drift_template)
    #
    # fig.show()
