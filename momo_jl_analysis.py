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
# […]

# Own modules
from joblib import load
import cv2
from matplotlib import pylab


def draw_contour(ch=None, ch_name=None,channel='phase',time=0, fov_jl=None):
    """
    draw contours in chamber for checking the images' segmentation.
    :param ch: int, chamber index
    :param ch_name: str, chamber name
    :param channel: str, channel color
    :param time: int or list.
    :param fov_jl: memory data
    :return: channel image with cell contour.
    """
    if ch:
        ch_na = fov_jl['chamber_loaded_name'][ch]
        # ch_index = list(range(len(fov_jl['chamber_loaded_name'])))[ch]
    else:
        ch_na = ch_name

    channl_key = dict(phase='chamber_phase_images',
                      green='chamber_green_images',
                      red='chamber_red_images')
    channl_color = channl_key[channel]

    if not isinstance(time, int):
        time = slice(*time)
        channel_im = fov_jl[channl_color][ch_na][time]
    else:
        channel_im = fov_jl[channl_color][ch_na][time]
        channel_im = np.expand_dims(channel_im, axis=0)

    if channel == 'phase':
        cell_cuntour = fov_jl['chamber_cells_contour'][ch_na][time]
    else:
        time_str = fov_jl['times'][channel][time]
        if isinstance(time_str, str):
            time_index = fov_jl['times']['phase'].index(time_str)
            cell_cuntour = [fov_jl['chamber_cells_contour'][ch_na][time_index]]

        else:
            time_index = [fov_jl['times']['phase'].index(ele) for ele in time_str]
            cell_cuntour = []
            for inx in time_index:
                cell_cuntour.append(fov_jl['chamber_cells_contour'][ch_na][inx])

    ims_with_cnt = []
    for i, cts in enumerate(cell_cuntour):
        ims_with_cnt.append(
            cv2.drawContours(to_BGR(rangescale(channel_im[i], (0, 255)).astype(np.uint8)), cts, -1,
                             (247, 220, 111),
                             1))
    ims_with_cnt = np.concatenate(ims_with_cnt, axis=1)
    return ims_with_cnt


def find_jl(dir):
    fn = [f for f in os.listdir(dir) if f.split('.')[-1] == 'jl']
    fn = [os.path.join(dir, f) for f in fn]
    return fn

def convert_time(time):
    h, s = divmod(time, 60 * 60)
    m, s = divmod(s, 60)
    h = h + m / 60 + s / 3600
    return h

def to_BGR(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def rangescale(frame, rescale):
    '''
    Rescale image values to be within range

    Parameters
    ----------
    frame : 2D numpy array of uint8/uint16/float/bool
        Input image.
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    '''
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame - np.min(frame)) / np.ptp(frame)) * np.ptp(rescale) + rescale[0]
    else:
        frame = np.ones_like(frame) * (rescale[0] + rescale[1]) / 2
    return frame

#%%
DIR = r'D:\python_code\MoMa_process\test_data_set\jl_data'
jl_file = find_jl(DIR)
fov_jl = load(jl_file[-1])

#%%
ims_with_cnt = draw_contour(ch=4, channel='gree', time=[200, 209], fov_jl=fov_jl)
pylab.imshow(ims_with_cnt)
pylab.show()
