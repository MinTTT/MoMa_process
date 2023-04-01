"""
 This file used to extract cells' profile in nd2 file.
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports

# =========== CUP only===================== #
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# =========== CUP only===================== #

import pandas as pd
from utils.delta.model import unet_seg
from utils.delta.data import postprocess
import matplotlib.pyplot as plt
import tensorflow as tf
import tifffile as tiff
import cv2
from utils.delta.utilities import rangescale
from tqdm import tqdm
import numpy as np  # Or any other
from typing import Tuple, Union, Dict, Optional, Any
from scipy.stats import binned_statistic
from warnings import warn
import utils.sciplot as splt
from scipy.optimize import leastsq, minimize

splt.whitegrid()

# Allow memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# %%
def cell_segmentation(imdir=None,
                      model_seg=None,
                      split_factor=8, debug=False, overlap=0.05, **kwargs):
    """
    This function used to predict cell pixels in a given image.
    :param split_factor:
    :type split_factor:
    :param overlap:
    :type overlap:
    :param debug:
    :type debug:
    :param imdir: str. image dir, input a 2048X2048 image.
    :param model_seg: predict model_for_colony.
    :return: nd array, a binary image.
    """
    if 'im' in kwargs:
        im = kwargs['im']
    else:
        im = tiff.imread(imdir).squeeze()
    im = rangescale(im, (0, 1.))

    # ==========split image into 8X8 sub-images.=================
    img0Len, im1Len = im.shape
    overlap_len = int(overlap * img0Len / split_factor)
    if overlap_len % 2 != 0:
        overlap_len += 1
    half_overlap_len = int(overlap_len / 2)

    splitindex = np.arange(0, img0Len + 1, int(img0Len / split_factor)).astype(int)
    indexs = []
    for i in range(split_factor):
        for j in range(split_factor):
            a = splitindex[i] - half_overlap_len
            b = splitindex[i + 1] + half_overlap_len
            c = splitindex[j] - half_overlap_len
            d = splitindex[j + 1] + half_overlap_len
            if a < 0:
                b += half_overlap_len
                a = 0
            if b > img0Len:
                b = img0Len
                a -= half_overlap_len
            if c < 0:
                c = 0
                d += half_overlap_len
            if d > im1Len:
                d = im1Len
                c -= half_overlap_len
            indexs.append(((a, b), (c, d)))

    ims = np.zeros((img0Len, im1Len))
    subimg = np.empty((len(indexs),) + (int(im1Len / split_factor), int(img0Len / split_factor)))
    for indexSlice, imSlice in enumerate(indexs):
        # print(imSlice)
        a, b, c, d = imSlice[0][0], imSlice[0][1], imSlice[1][0], imSlice[1][1]
        subimg[indexSlice, ...] = cv2.resize(im[a:b, c:d], (int(im1Len / split_factor), int(img0Len / split_factor)))

    seg_inputs = np.expand_dims(subimg, axis=3)
    seg = model_seg.predict(seg_inputs)
    seg = seg.squeeze()  # remove axis3
    for indexSlice, imSlice in enumerate(indexs):
        a, b, c, d = imSlice[0][0], imSlice[0][1], imSlice[1][0], imSlice[1][1]
        subseg = cv2.resize(seg[indexSlice, ...], (d - c, b - a))
        mask = subseg > ims[a:b, c:d]
        ims[a:b, c:d][mask] = subseg[mask]

    if 'square_size' in kwargs:
        seg = postprocess(ims, square_size=kwargs['square_size'])
    elif 'min_size' in kwargs:
        seg = postprocess(ims, min_size=kwargs['min_size'])
    else:
        seg = postprocess(ims, square_size=10, min_size=300, iterNum=1)
    if debug:
        return ims, seg
    return seg


def back_corrt(im: np.ndarray, bac: float) -> np.ndarray:
    im -= bac
    im[im < 0] = 0.
    return im


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


def plotprediction(ims, singleSize=15):
    if isinstance(ims, list):
        imsNum = len(ims)
        fig, axs = plt.subplots(1, imsNum, figsize=(imsNum * singleSize * 1.1, singleSize))
    else:
        imsNum = 1
        ims = [ims]
        fig, axs = plt.subplots(1, imsNum, figsize=(imsNum * singleSize * 1.1, singleSize))
        axs = [axs]

    for i, im in enumerate(ims):
        axs[i].imshow(im, cmap='gray')

    fig.show()
    return None


def cv_otsu(images: np.ndarray, gaussian_core=(3, 3)) -> Tuple[np.ndarray, list]:
    '''

    Parameters
    ----------
    images :
    gaussian_core :

    Returns
    -------

    '''
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
    """

    Parameters
    ----------
    cnt : np.ndarray
        contour list, this contour list is openCV-like.
    size : tuple
        image size, python-like (0 axis length, 1 axis length)

    Returns
    -------
    tuple
        (mask-image, pixels-index)

        mask-image : np.ndarray

        pixels-index: np.ndarray
            [0 axis index, 1 axis index]

    """
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


from scipy.optimize import fmin_bfgs


class TwodegreePoly(object):
    a = None
    b = None
    c = None

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x * x + self.b * x + self.c

    def _derivative(self, x: Union[float, np.ndarray]):
        return self.a * 2. * x + self.b

    def derivative(self):
        return self._derivative


# def FitCellParameters(newPars, cellMask, x_p, y_p, a, b, c, cell_width):
#     # if isinstance(newPars, list):
#     #     newPars = np.array(newPars)
#     # newPars = newPars / 1000
#     def calculate_Chi(newPars, cellMask, x_p, y_p, a, b, c, cell_width):
#         # newPars = newPars * 1000
#         x_min, x_max = newPars
#
#         x_c = calc_xc([c, b, a], x_p, y_p)
#         x_c[x_c < x_min] = x_min
#         x_c[x_c > x_max] = x_max
#
#         r = calc_r([a, b, c], x_p, y_p, x_c)
#
#         newmask = np.zeros(r.shape, dtype=bool)
#         newmask[r <= (cell_width / 2)] = True
#         chi2 = np.sum(newmask ^ cellMask).astype(float)
#         return chi2
#
#     fit_parameters = minimize(calculate_Chi, newPars, args=(cellMask, x_p, y_p, a, b, c, cell_width),
#                               method='Nelder-Mead')
#     return fit_parameters

def calculate_Chi(x_min, x_max, cellMask, x_p, y_p, a, b, c, cell_width):
    # newPars = newPars * 1000
    # x_min, x_max = newPars

    x_c = calc_xc([c, b, a], x_p, y_p)
    x_c[x_c < x_min] = x_min
    x_c[x_c > x_max] = x_max

    r = calc_r([a, b, c], x_p, y_p, x_c)

    newmask = np.zeros(r.shape, dtype=bool)
    newmask[r <= (cell_width / 2)] = True
    chi2 = np.sum(newmask ^ cellMask).astype(float)
    return chi2


def FitCellParameters(x_min, x_max, cellMask, x_p, y_p, a, b, c, cell_width,
                      modes=None):
    def opt_xir(newPars, cellMask, x_p, y_p, a, b, c, cell_width):
        x_min, x_max = newPars
        return calculate_Chi(x_min, x_max, cellMask, x_p, y_p, a, b, c, cell_width)

    def opt_poly2(newPars, cellMask, x_p, y_p, cell_width, x_min, x_max):
        a, b, c = newPars
        return calculate_Chi(x_min, x_max, cellMask, x_p, y_p, a, b, c, cell_width)

    def opt_width(newPars, cellMask, x_p, y_p, x_min, x_max):
        cell_width = newPars
        return calculate_Chi(x_min, x_max, cellMask, x_p, y_p, a, b, c, cell_width)

    if modes == 'opt_xir':
        fit_parameters = minimize(opt_xir, np.array([x_min, x_max]),
                                  args=(cellMask, x_p, y_p, a, b, c, cell_width),
                                  method='Nelder-Mead')
    elif modes == 'opt_width':
        fit_parameters = minimize(opt_width, np.array([cell_width]),
                                  args=(cellMask, x_p, y_p, x_min, x_max),
                                  method='Nelder-Mead')
    elif modes == 'opt_poly2':
        fit_parameters = minimize(opt_poly2, np.array([a, b, c]),
                                  args=(cellMask, x_p, y_p, cell_width, x_min, x_max),
                                  method='Nelder-Mead')
    else:
        raise Warning('modes should be opt_xir, opt_wideth, or opt_poly2')
        # return None

    return fit_parameters


def calc_r(pars, x_p, y_p, x_c):
    """
    Calculate r_c in cell coord. r_c = sqrt(( x_c - x_p)^2 - (y_c - y_p)^2 )
    Parameters
    ----------
    pars : array-like
        polynomial parameters of cell coord. [a, b, c]
    x_p : array-like
        x_p of cell coord.
    y_p : array-like
        y_p of cell coord.
    x_c :
        y_c of cell coord.

    Returns
    -------
    array: array-like
        the distance between cell spine to (x_p, y_p)

    """
    a, b, c = pars
    y_c = a * x_c ** 2 + b * x_c + c
    r = np.sqrt((x_c - x_p) ** 2 + (y_c - y_p) ** 2)
    return r


def FitTwoDegreePoly(x, y, initPars=None):
    if initPars is None:
        initPars = [1, 1, 1]

    def TwoDegErro(pars):
        a, b, c = pars
        return y - a * x * x - b * x - c

    # opt_ret = fmin_bfgs(TwoDegErro, initPars, disp=False)
    opt_ret = leastsq(TwoDegErro, initPars)[0]
    polyline = TwodegreePoly(*opt_ret)
    return polyline


def solve_general(a, b, c, d):
    """
    Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d.

    Only works if polynomial discriminant < 0, then there is only one real root which is the one that is returned. [1]_


    Parameters
    ----------
    a : array_like
        Third order polynomial coefficient.
    b : array_like
        Second order polynomial coefficient.
    c : array_like
        First order polynomial coefficient.
    d : array_like
        Zeroth order polynomial coefficient.

    Returns
    -------
    array : array_like
        Real root solution.

    .. [1] https://en.wikipedia.org/wiki/Cubic_function#General_formula

    """

    # todo check type for performance gain?
    # 16 16: 5.03 s
    # 32 32: 3.969 s
    # 64 64: 5.804 s
    # 8 8:
    d0 = b ** 2. - 3. * a * c
    d1 = 2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d

    r0 = np.square(d1) - 4. * d0 ** 3.
    r1 = (d1 + np.sqrt(r0)) / 2
    dc = np.cbrt(
        r1)  # power (1/3) gives nan's for coeffs [1.98537881e+01, 1.44894594e-02, 2.38096700e+00]01, 1.44894594e-02, 2.38096700e+00]
    return -(1. / (3. * a)) * (b + dc + (d0 / dc))
    # todo hit a runtimewaring divide by zero on line above once


def solve_trig(a, b, c, d):
    """
    Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
    Only for polynomial discriminant > 0, the polynomial has three real roots [1]_

    Parameters
    ----------
    a : array_like
        Third order polynomial coefficient.
    b : array_like
        Second order polynomial coefficient.
    c : array_like
        First order polynomial coefficient.
    d : array_like
        Zeroth order polynomial coefficient.

    Returns
    -------
    array : array_like
        First real root solution.

    .. [1] https://en.wikipedia.org/wiki/Cubic_function#Trigonometric_solution_for_three_real_roots

    """

    p = (3. * a * c - b ** 2.) / (3. * a ** 2.)
    q = (2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d) / (27. * a ** 3.)
    assert (np.all(p < 0))
    k = 0.
    t_k = 2. * np.sqrt(-p / 3.) * np.cos(
        (1 / 3.) * np.arccos(((3. * q) / (2. * p)) * np.sqrt(-3. / p)) - (2 * np.pi * k) / 3.)
    x_r = t_k - (b / (3 * a))
    try:
        assert (np.all(
            x_r > 0))  # don't know if this is guaranteed otherwise boundaries need to be passed and choosing from 3 
        # slns
    except AssertionError:
        pass
        # todo find out if this is bad or not
        # raise ValueError
    return x_r


def calc_xc(coeff, xp, yp):
    """
    Calculates the coordinate xc on p(x) closest to xp, yp.

    All coordinates are cartesian. Solutions are found by solving the cubic equation.

    Parameters
    ----------
    coeff: arry-like
        a list or tuple containing coefficients of
    xp : :obj:`float` or :class:`~numpy.ndarray`
        Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
    yp : :obj:`float` : or :class:`~numpy.ndarray`
        Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

    Returns
    -------
    xc : :obj:`float` or :class:`~numpy.ndarray`
        Cellular x-coordinate for point(s) xp, yp
    """

    assert xp.shape == yp.shape
    # https://en.wikipedia.org/wiki/Cubic_function#Algebraic_solution
    a0, a1, a2 = coeff
    # xp, yp = xp.astype('float32'), yp.astype('float32')
    # Converting of cell spine polynomial coefficients to coefficients of polynomial giving distance r
    a, b, c, d = 4 * a2 ** 2, \
                 6 * a1 * a2, \
                 4 * a0 * a2 + 2 * a1 ** 2 - 4 * a2 * yp + 2, \
                 2 * a0 * a1 - 2 * a1 * yp - 2 * xp
    # a: float, b: float, c: array, d: array
    discr = 18 * a * b * c * d - 4 * b ** 3 * d + b ** 2 * c ** 2 - 4 * a * c ** 3 - 27 * a ** 2 * d ** 2

    # if np.any(discr == 0):
    #     raise ValueError('Discriminant equal to zero encountered. This should never happen. Please make an issue.')

    if np.all(discr < 0):
        x_c = solve_general(a, b, c, d)
    else:
        x_c = np.zeros(xp.shape)
        mask = discr < 0

        general_part = solve_general(a, b, c[mask], d[mask])
        trig_part = solve_trig(a, b, c[~mask], d[~mask])

        x_c[mask] = general_part
        x_c[~mask] = trig_part

    return x_c


class Cell:
    def __init__(self, cell_index=None, fov=None, umppx=0.065):
        self.cell_index = cell_index  # type: list  # [chamber_key, time_index, cell_index]
        self.channels = []
        self.fov = fov
        self.fov_size = None
        self.contour_init = None
        self._contour_opt = None
        self.skeleton = None
        self.edge_mask = None
        self.edge_mask_xy = None
        self.skeleton_func = None
        self.spine = None
        self.spine_length = None
        self.area = None
        self.umppx = umppx  # type: float  # nu m per pixel
        self.channel_imgs = {}  # type: Optional[Dict[str, np.ndarray]]
        self.flu_level = {}  # type: Optional[Dict[str, Any]]
        self.rectangle = None  # type: Optional[dict]
        self.verticalCell = None
        self.width = None
        self.width_area = None
        self.intensity = None
        self.x_min = None
        self.x_max = None
        self.snapSlice = []

    def set_mask_init(self, cnt, size):
        self.contour_init = cnt
        self.fov_size = size
        self._contour_opt = self.contour_init.copy()

    @property
    def mask_opt(self):
        mask, _ = cv_full_contour2mask(self._contour_opt, self.fov_size)
        return mask

    @mask_opt.setter
    def mask_opt(self, img):
        contour_opt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._contour_opt = contour_opt[0]

    @property
    def mask(self):
        mask, _ = cv_full_contour2mask(self._contour_opt, self.fov_size)

        return mask

    @property
    def contour(self):
        return self._contour_opt

    def cal_cell_skeleton(self):

        cellmask = self.mask_opt
        edge_mask, edge_mask_xy = cv_edge_fullmask2contour(cellmask)

        edge0range, edge1range = np.ptp(edge_mask_xy[:, 0]), np.ptp(edge_mask_xy[:, 1])
        # cellmask = self.mask_opt
        # whether cell was vertical placed in fov ?
        if edge1range <= edge0range:  # yes, vertical
            bins = int(edge1range / 3)
            self.verticalCell = True
        else:  # No, horizontal.
            bins = int(edge0range / 3)
            self.verticalCell = False

        if bins < 5:
            bins = 5

        if self.verticalCell:
            xindex = 0
            yindex = 1
        else:
            xindex = 1
            yindex = 0

        erosion = cv_erode(cellmask, 6, 2)

        skelpoints = np.where(erosion)  # (0 index, 1 index)
        # print(skelpoints)
        if len(skelpoints[0]) == 0:
            return None
        y_stat = binned_statistic(skelpoints[xindex], skelpoints[yindex],
                                  'mean', bins=bins)
        # print(y_stat)
        x_stat = np.diff(y_stat[-2]) / 2 + y_stat[-2][:-1]

        self.skeleton = np.hstack((x_stat.reshape(-1, 1), y_stat[0].reshape(-1, 1)))
        # self.skeleton = np.hstack((skelpoints[xindex].reshape(-1, 1), skelpoints[yindex].reshape(-1, 1)))

        # self.skeleton_func = UnivariateSpline(self.skeleton[:, 0], self.skeleton[:, 1], k=2)
        self.skeleton_func = FitTwoDegreePoly(self.skeleton[:, 0], self.skeleton[:, 1])

        self.x_min, self.x_max = self._contour_opt[:, 0, yindex].min(), self._contour_opt[:, 0, yindex].max()
        x_space = np.linspace(self.x_min, self.x_max, endpoint=True)
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

        dists = []
        for indexsket in range(len(self.spine)):
            dist = ((self.contour[:, 0, yindex] - self.spine[indexsket, 0]) ** 2 + (
                    self.contour[:, 0, xindex] - self.spine[indexsket, 1]) ** 2) ** (1 / 2)
            # print(dist)
            dists.append(np.nanmin(dist))
        self.width = np.nanmedian(dists) * self.umppx * 2

        a, b, c = (np.pi - 4) / 4, self.spine_length, -self.area
        self.width_area = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        self.intensity = None

    def opt_skeleton(self):
        pixle1, pixle0 = np.meshgrid(np.arange(self.fov_size[0], dtype=int),
                                     np.arange(self.fov_size[1], dtype=int))

        self.getCellSnapRange()
        contoursnap = self.mask_opt[self.snapSlice[0], self.snapSlice[1]]
        # len0, len1 = contoursnap.shape
        subPixle1 = pixle1[self.snapSlice[0], self.snapSlice[1]]
        subPixle0 = pixle0[self.snapSlice[0], self.snapSlice[1]]
        if self.verticalCell:
            x_p, y_p = subPixle0, subPixle1
        else:
            x_p, y_p = subPixle1, subPixle0

        a, b, c = self.skeleton_func.a, self.skeleton_func.b, self.skeleton_func.c

        ret_xir = FitCellParameters(self.x_min, self.x_max,
                                    contoursnap.astype(bool),
                                    x_p, y_p, a, b, c, self.width_area / self.umppx,
                                    'opt_xir')

        ret_abc = FitCellParameters(*ret_xir.x,
                                    contoursnap.astype(bool),
                                    x_p, y_p, a, b, c, self.width_area / self.umppx,
                                    'opt_poly2')

        ret_width = FitCellParameters(*ret_xir.x,
                                      contoursnap.astype(bool),
                                      x_p, y_p, *ret_abc.x, self.width_area / self.umppx,
                                      'opt_width')
        self.skeleton_func = 


    def getCellSnapRange(self, snapSize=64):
        cellLoc = self.contour[0, 0, :][::-1]
        snapUL = cellLoc - snapSize
        snapDR = cellLoc + snapSize
        snapUL[snapUL < 0] = 0
        snapDR[snapDR > yLen] = yLen
        self.snapSlice = [slice(snapUL[0], snapDR[0]), slice(snapUL[1], snapDR[1])]

    def assign_channel_img(self, channel_name: str, channel_img: np.ndarray):
        if channel_name in self.channels:
            warn(f"{self.cell_index[0]} {self.cell_index[1]} {self.cell_index[2]}: {channel_name} has "
                 f"already assigned, the new imported image will cover the old one.")
        self.channel_imgs[channel_name] = channel_img
        self.channels.append(channel_name)

    def reassign_channel_imgs(self, channel_imgs: Dict[str, np.ndarray]):
        self.channel_imgs = channel_imgs
        self.channels = list(channel_imgs.keys())

    def cal_cel_flu_level(self, channels: Union[str, list]):
        if isinstance(channels, str):
            channels = [channels]
        channels = [channel for channel in channels if channel in self.channels]
        if channels:
            for channel in channels:
                flu_img = self.channel_imgs[channel]
                flu_pixels = flu_img[self.mask_opt != 0]
                flu_avg = np.mean(flu_pixels)
                flu_medium = np.quantile(flu_pixels, 0.95)
                self.flu_level[channel] = dict(mean=flu_avg, medium=flu_medium)


# […]
MODEL_FILE = r'./test_data_set/model/delta_pads_seg.hdf5'

# %% ND2 reader

tiffPath = r'K:\AD_data\MOPS_Gly_100.tif'

imageData = tiff.imread(tiffPath)
imageNum, xLen, yLen = imageData.shape
target_size_seg = (1024, 1024)
model_seg = unet_seg(input_size=target_size_seg + (1,))
model_seg.load_weights(MODEL_FILE)

segmentations = np.zeros((imageNum, xLen, yLen), dtype=np.uint8)

for i in tqdm(range(imageNum)):
    segs = cell_segmentation(model_seg=model_seg, im=imageData[i, ...], debug=False, split_factor=2, overlap=0.01)
    segmentations[i, ...] = segs.astype(np.uint8)

tiff.imsave(tiffPath + '.seg.tif', segmentations)

# %%
tiffPath = r'K:\AD_data\MOPS_Gly_100.tif'
contourpath = tiffPath + '.seg.tif'
imageData = tiff.imread(tiffPath)
contourData = tiff.imread(contourpath)
imageNum, xLen, yLen = imageData.shape
imageNum = 1
snapSize = (64, 64)
maxium_iter = 3
print('========> Processing contours')
fovCells = []
for fovIndex in tqdm(range(imageNum)):
    contours = cv2.findContours(contourData[fovIndex, ...], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # find
    contours.sort(key=lambda elem: np.sum(np.max(elem[:, 0, 0] + np.max(elem[:, 0, 1]))))
    cells = []
    for c, contour in enumerate(contours):  # Run through cells in frame
        cell = Cell(cell_index=c + 1, fov=fovIndex + 1)
        cell.set_mask_init(contour, size=(yLen, xLen))
        cell.assign_channel_img('phase', imageData[fovIndex, ...])
        cells.append(cell)
    fovCells.append(cells)
print('========> Revising contours')
for fovIndex in tqdm(range(imageNum)):
    cells = fovCells[fovIndex]
    cells_number = len(cells)
    iter_key = [True] * cells_number

    for i in range(maxium_iter):
        all_cell_mask = [c.mask_opt for c in cells]
        blankMask = np.sum(all_cell_mask, axis=0, dtype=bool)
        for index, cell in enumerate(cells):
            cellLoc = cell.contour[0, 0, :][::-1]
            snapUL = cellLoc - snapSize
            snapDR = cellLoc + snapSize
            snapUL[snapUL < 0] = 0
            snapDR[snapDR > yLen] = yLen
            snapSlice = [slice(snapUL[0], snapDR[0]), slice(snapUL[1], snapDR[1])]
            cellsnap = cell.channel_imgs['phase'][snapSlice[0], snapSlice[1]]

            cellmasksnap = all_cell_mask[index][snapSlice[0], snapSlice[1]]
            cellmasksnapOld = cellmasksnap.copy()
            othercellmask = np.logical_xor(blankMask[snapSlice[0], snapSlice[1]], cellmasksnapOld).astype(np.uint8)
            convertsnap = 1 - rangescale(cellsnap, (0, 1.))

            snapmask_new, otsu_thre = cv_otsu(convertsnap)
            cell_mask_edge, mask_edge_xy = cv_edge_fullmask2contour(cellmasksnap, thickness=8)
            edge_revise_mask = np.logical_or(cellmasksnap, cell_mask_edge)
            snapmask_new = np.logical_and(edge_revise_mask, snapmask_new)
            snapmask_new = cv_open(snapmask_new.astype(np.uint8))
            snapmask_new = np.logical_and(snapmask_new, np.logical_not(othercellmask))

            if snapmask_new.any():
                mask_ = np.zeros((yLen, xLen), dtype=np.uint8)
                mask_[snapSlice[0], snapSlice[1]] = snapmask_new.astype(np.uint8)
                cell.mask_opt = mask_
                checkarray = cellmasksnapOld == snapmask_new
                if checkarray.all():
                    iter_key[index] = False
            else:
                iter_key[index] = False
            if True not in iter_key:
                break

# Calculating call parameters
cells_length = []
cells_width = []
cells_width_area = []
cells_area = []
fovs = []
cellindex = []
cells_intensity = []
cellX = []
cellY = []
for fovIndex in tqdm(range(imageNum)):
    cells = fovCells[fovIndex]
    cells_number = len(cells)

    for cell in cells:  # type: Cell
        # cell.set_mask_init(contour, size=(yLen, xLen))
        cell.cal_cell_skeleton()
        cell.cal_cel_flu_level('phase')
        # cells.append(cell)
        cells_length.append(cell.spine_length)
        cells_width.append(cell.width)
        cells_area.append(cell.area)
        cells_width_area.append(cell.width_area)
        cellindex.append(cell.cell_index)
        fovs.append(cell.fov)
        cells_intensity.append(cell.flu_level['phase']['medium'])
        cellX.append(cell.contour[0, 0, 0])
        cellY.append(cell.contour[0, 0, 1])

# %%  optimize mask

if cell.verticalCell:
    xindex = 0
    yindex = 1
else:
    xindex = 1
    yindex = 0
pixle1, pixle0 = np.meshgrid(np.arange(cell.fov_size[0], dtype=int),
                             np.arange(cell.fov_size[1], dtype=int))

cell.getCellSnapRange()

contoursnap = cell.mask_opt[cell.snapSlice[0], cell.snapSlice[1]]
len0, len1 = contoursnap.shape
subPixle1 = pixle1[cell.snapSlice[0], cell.snapSlice[1]]
subPixle0 = pixle0[cell.snapSlice[0], cell.snapSlice[1]]
if cell.verticalCell:
    x_p, y_p = subPixle0, subPixle1
    # x_p, y_p = pixle1[:len0, :len1], subPixle0[:len0, :len1]

    xP, yP = cell.snapSlice[1].start, cell.snapSlice[0].start

else:
    # x_p, y_p = pixle1[:len0, :len1], pixle1[:len0, :len1]
    x_p, y_p = subPixle1, subPixle0
    xP, yP = cell.snapSlice[0].start, cell.snapSlice[1].start

a, b, c = cell.skeleton_func.a, cell.skeleton_func.b, cell.skeleton_func.c

ret_xir = FitCellParameters(cell.x_min, cell.x_max,
                            contoursnap.astype(bool),
                            x_p, y_p, a, b, c, cell.width_area / cell.umppx,
                            'opt_xir')

ret_abc = FitCellParameters(*ret_xir.x,
                            contoursnap.astype(bool),
                            x_p, y_p, a, b, c, cell.width_area / cell.umppx,
                            'opt_poly2')

ret_width = FitCellParameters(*ret_xir.x,
                              contoursnap.astype(bool),
                              x_p, y_p, *ret_abc.x, cell.width_area / cell.umppx,
                              'opt_width')

plotprediction([imageData[0, ...][cell.snapSlice[0], cell.snapSlice[1]],
                contoursnap,
                chi2])

# %%
#
df = pd.DataFrame(data=dict(cellLength=cells_length,
                            cellWidth=cells_width,
                            cellWidthArea=cells_width_area,
                            cellArea=cells_area,
                            cellIntensity=cells_intensity,
                            cellFove=fovs,
                            cellIndex=cellindex,
                            cellX=cellX,
                            cellY=cellY
                            ))

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.imshow(imageData[0, ...], cmap='gray')
for cell in cells[-2:]:
    if cell.spine is not None:
        if cell.verticalCell:
            ax.plot(cell.spine[:, 1], cell.spine[:, 0], lw=1)
        else:
            ax.plot(cell.spine[:, 0], cell.spine[:, 1], lw=1)

ax.set_xlim(0, 2048)
ax.set_ylim(0, 2048)

fig.show()
# %%
from sklearn.decomposition import PCA
from sklearn import manifold
from scipy.stats import gaussian_kde

# import utils.sciplot as splt

# splt.whitegrid()
dataMask = []
for lenIndex in range(len(df)):
    boolData = ~np.isnan(df)
    if boolData.iloc[lenIndex].all():
        dataMask.append(True)
    else:
        dataMask.append(False)

cleanedCell = df[dataMask]
data_set = np.array(cleanedCell[['cellLength', 'cellWidth', 'cellWidthArea', 'cellArea', 'cellIntensity']])

data_set = data_set / np.max(data_set, axis=0)

cellPCA = PCA()
cellPCA.fit(data_set)

transdata = cellPCA.transform(data_set)

z_pca = gaussian_kde(transdata.T[:2, :])(transdata.T[:2, :])

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax.scatter(transdata[:, 0], transdata[:, 1], c=z_pca, cmap='coolwarm'
           )
fig.show()
# fig.savefig(tiffPath + '.PCA.svg', bbox_inches='tight', transparent=True)

tsne = manifold.TSNE(n_components=2, random_state=0)
trans_data_tSNE = tsne.fit_transform(data_set)
ztSNE = gaussian_kde(trans_data_tSNE.T)(trans_data_tSNE.T)

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax.scatter(trans_data_tSNE[:, 0], trans_data_tSNE[:, 1], c=z_pca, cmap='coolwarm'
           )
fig.show()
# fig.savefig(tiffPath + '.tSNE.svg', bbox_inches='tight', transparent=True)

tsneData = pd.DataFrame(data=np.hstack([trans_data_tSNE, z_pca.reshape(-1, 1)]))
# tsneData.to_csv(tiffPath + '.tSNEdata.csv')

# %% After data selection by lasso

lassoData = pd.read_csv(tiffPath + '.tSNEdata.csv.lasso.csv')
lassoindex = lassoData.iloc[:, 1]
lassoCells = cleanedCell.iloc[np.array(lassoindex).astype('int'), :]
lassoCells.to_csv(tiffPath + '.tSNEdata.csv.lasso.celldata.csv')
lassoCells.describe().to_csv(tiffPath + '.tSNEdata.csv.lasso.describe.csv')
