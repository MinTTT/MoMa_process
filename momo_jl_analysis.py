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
import matplotlib.pyplot as plt
from tqdm import tqdm

# import dask
# from dask.distributed import Client
# client = Client(n_workers=16, threads_per_worker=32)


GREEN_COLOR = (0, 255, 0)  # RGB
RED_COLOR = (255, 0, 0)  # RGB


def draw_contour(ch=None, ch_name=None,
                 channel='phase', time=0, fov_jl=None,
                 contours=True, color=None, threshold=None, conat=True):
    """
    draw contours in chamber for checking the images' segmentation.
    :param conat: default True, if False, return a list of chambers
    :param threshold: color threshold
    :param color: tuple, RGB color
    :param contours: default True, draw cells contour
    :param ch: int, chamber index
    :param ch_name: str, chamber name
    :param channel: str, channel color
    :param time: int or list.
    :param fov_jl: memory data
    :return: channel image with cell contour.
    """
    if ch is not None:
        ch_na = fov_jl['chamber_loaded_name'][ch]
    else:
        ch_na = ch_name
    # print(ch_na)

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
        thread_im = rangescale(channel_im[i], (0, 255), threshold).astype(np.uint8)
        bgr_im = to_BGR(thread_im)
        if color:
            bgr_im = map_color(bgr_im, color)
        if contours:
            ims_with_cnt.append(
                cv2.drawContours(bgr_im, cts, -1,
                                 (247, 220, 111),
                                 1))
        else:
            ims_with_cnt.append(bgr_im)
    if conat:
        ims_with_cnt = np.concatenate(ims_with_cnt, axis=1)
    return ims_with_cnt


def map_color(im: np.ndarray, color: tuple) -> np.ndarray:
    for i in range(3):
        im[..., i] = rangescale(im[..., i], (0, color[i]), threshold=color[i]).astype(np.uint8)
    return im


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


def rangescale(frame, rescale, threshold=None) -> np.ndarray:
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
    if threshold:
        scale = threshold / max(rescale)
        frame[frame > threshold] = threshold
        frame = frame / scale
        return frame
    if np.ptp(frame) > 0:
        frame = ((frame - np.min(frame)) / np.ptp(frame)) * np.ptp(rescale) + rescale[0]
    else:
        frame = np.ones_like(frame) * (rescale[0] + rescale[1]) / 2
    return frame


# %%
DIR = r'./test_data_set/test_data'
jl_file = find_jl(DIR)
fov_jl = load(jl_file[0])

# %%
from utils.signal import vertical_mean
from scipy.fftpack import fft, fftfreq, ifft, fftshift


def image_conv(imas):
    chamber_im = imas
    chose_index = np.random.choice(len(chamber_im), 10)
    len_im = chamber_im.shape[-2]
    v_means = []
    for i in chose_index:
        v_means.append(vertical_mean(chamber_im[i]))

    conv_sg = np.ones(len_im)
    for vm in v_means:
        vm = vm - np.mean(vm)
        conv_sg = conv_sg * fft(vm)

    freq = fftfreq(len_im)
    mask = freq > 0
    return conv_sg[mask], freq[mask]


ims_chamber = fov_jl['chamber_phase_images']
chamber_names = list(ims_chamber.keys())
ch1 = 0

fig1, ax = plt.subplots(1, 1)
ax.imshow(ims_chamber[chamber_names[ch1]][0])

fig1.show()

# chamber_im = ims_chamber[chamber_names[9]]
fig1, ax = plt.subplots(1, 1)
# ax.imshow(chamber_im[0])
for name in chamber_names:
    sg, frq = image_conv(ims_chamber[name])
    ax.plot(frq, np.log(abs(sg)))
# ax.plot(freq[mask], abs(sg[mask]))
fig1.show()
#%%
# lineage linking



#%%
ch = 3
time = [0, -1]

ims_with_cnt = draw_contour(ch=ch, channel='phase', time=time, fov_jl=fov_jl)

fig1, ax = plt.subplots(1, 1)
ax.imshow(rangescale(ims_with_cnt, (0, 255)).astype(np.uint8))
ax.grid(False)
fig1.show()

# %%  show specific fluorescent channels along time
ch = 4
time = [0, -1]
ims_with_cnt_green = draw_contour(ch=ch, channel='green', time=time, fov_jl=fov_jl,
                                  color=GREEN_COLOR, threshold=600, contours=False)
ims_with_cnt_red = draw_contour(ch=ch, channel='red', time=time, fov_jl=fov_jl,
                                color=RED_COLOR, threshold=800, contours=False)
# ims_with_cnt = [ims_with_cnt_green[i] + ims_with_cnt_red[i] for i in range(len(ims_with_cnt_red))]
ims_with_cnt = ims_with_cnt_green ** 0.32 + ims_with_cnt_red ** 0.32  # nonlinear rescale
fig1, ax = plt.subplots(1, 1)
ax.imshow(rangescale(ims_with_cnt, (0, 255)).astype(np.uint8))
ax.grid(False)
fig1.show()
# %% animation of channel along time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ch = 3
ims_with_cnt_green = draw_contour(ch=ch, channel='green', time=[0, -1, 4], fov_jl=fov_jl,
                                  color=GREEN_COLOR, threshold=250, contours=False, conat=False)
ims_with_cnt_red = draw_contour(ch=ch, channel='red', time=[0, -1, 4], fov_jl=fov_jl,
                                color=RED_COLOR, threshold=200, contours=False, conat=False)
ims_with_cnt = [ims_with_cnt_green[i] + ims_with_cnt_red[i] for i in range(len(ims_with_cnt_red))]

fig2, ax2 = plt.subplots(1, 1, figsize=(25, 50))
ims = []
for i in range(len(ims_with_cnt)):
    ims.append([ax2.imshow(ims_with_cnt[i], animated=True)])

ax2.grid(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True,
                                repeat_delay=1000)
ani.save(os.path.join(DIR, f'ch_{ch}.avi'))
fig2.show()
