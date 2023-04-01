# Built-in/Generic Imports
import os

import numpy as np  # Or any other

from typing import Tuple, Union, Dict, List, Optional, Any
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
from warnings import warn

import cv2

from utils.delta.utilities import getChamberBoxes, getDriftTemplate, driftcorr, rangescale, cropbox


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


def cv_erode(img, kernel_size=3, itr=1):
    kernel = np.ones((kernel_size, kernel_size), np.int8)
    return cv2.erode(img, kernel, itr)


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
        self.verticalCell = None

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
        edge0range, edge1range = np.ptp(self.edge_mask_xy[:, 0]), np.ptp(self.edge_mask_xy[:, 1])

        # whether cell was vertical placed in fov ?
        if edge1range <= edge0range:  # yes, vertical
            bins = int(edge1range / 3)
            self.verticalCell = True
        else:  # No, horizontal.
            bins = int(edge0range / 3)
            self.verticalCell = False

        # bins = int(np.ptp(self.edge_mask_xy[:, 0]) / 3)
        if bins < 5:
            bins = 5

        if self.verticalCell:
            xindex = 0
            yindex = 1
        else:
            xindex = 1
            yindex = 0

        y_stat = binned_statistic(self.edge_mask_xy[:, xindex], self.edge_mask_xy[:, yindex],
                                  'mean', bins=bins)
        x_stat = np.diff(y_stat[-2]) / 2 + y_stat[-2][:-1]
        # self.skeleton = np.hstack((x_stat.reshape(-1, 1), y_stat[0].reshape(-1, 1)))
        self.skeleton = np.hstack((self.edge_mask_xy[:, xindex].reshape(-1, 1),
                                   self.edge_mask_xy[:, yindex].reshape(-1, 1)))
        self.skeleton_func = UnivariateSpline(self.skeleton[:, 0], self.skeleton[:, 1], k=3, s=None)
        x_min, x_max = self._contour_opt[:, xindex, 1].min(), self._contour_opt[:, xindex, 1].max()
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
