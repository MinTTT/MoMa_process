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
import tifffile as tiff
# […]

# Own modules
import momo_pipeline as mp
from utils.delta.utilities import cropbox
from matplotlib import pylab
import cv2
#%%
DIR = r'./test_data_set/test_data'

fovs_name = mp.get_fovs_name(DIR)
for i, fov in enumerate(fovs_name):
    print(f'Processing {i + 1}/{len(fovs_name)}')
    fov.process_flow()
#%%
fov1 = fovs_name[0]
fov1.detect_channels()
fov1.detect_frameshift()
fov1.cell_detection()
#%%

time, ch = 0, 5

ch_na = fov1.loaded_chamber_name[ch]
ch_index = fov1.index_of_loaded_chamber[ch]

im = tiff.imread(os.path.join(fov1.dir, fov1.fov_name, 'phase', fov1.times['phase'][time]))
if fov1.chamber_direction == 0:
    im = im[::-1, ...]
channel_box = fov1.chamberboxes[ch_index]
channel_im = cropbox(im, channel_box)
cell_cuntour = fov1.chamber_cells_contour[ch_na][time]
cell_contour_im = cv2.drawContours(channel_im, cell_cuntour)


pylab.imshow(fov1.chamber_cells_mask[ch_na][time])
pylab.show()

#%%
