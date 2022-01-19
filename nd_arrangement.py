# -*- coding: utf-8 -*-
'''
This file used to convert nd2 to tiff files with tvc arrangement
@author: CHU Pan
'''
# %%
import json
import os

# from nd2reader import ND2Reader
# from skimage import io as skio
import numpy as np
from joblib import Parallel, delayed
from tifffile import imsave
# import pims
from tqdm import tqdm

from nd2file import ND2MultiDim


def output_vof_tstack(images, file_name, save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass
    images.bundle_axes = 'yx'
    images.iter_axes = 'cvt'
    ims_t = images.imagecount
    ims_c = images.channels
    ims_v = images.multipointcount
    ims_y = images.height
    ims_x = images.width
    times = images.metadata['CustomData|AcqTimesCache']
    file_suffix = [f'c{c}_v{v}' for c in range(ims_c) for v in range(ims_v)]
    ims = []
    i = 0
    for index, fov in enumerate(images):
        ims.append(fov)
        print((index + 1) % ims_t)
        if ((index + 1) % ims_t == 0) and (index + 1 >= ims_t):
            im_name = save_dir + '\\' + file_name + '_' + file_suffix[i] + '.tif'
            i += 1
            # skio.imsave(im_name,
            #             np.array(ims).astype(np.uint16))
            ims = np.array(ims).astype(np.uint16)
            ims = ims.reshape(ims_t, 1, 1, ims_y, ims_x, 1)
            imsave(im_name, ims, imagej=True)
            print(f'file {im_name} saved!')
            ims = []

    # out put the nd2 file parameters
    josn_file = {'file_name': file_name,
                 'time_point': times.tolist(),
                 'metadata': images.metadata}
    josn_file = json.dumps(josn_file)
    file = open(save_dir + '\\' + file_name + '_time.json', 'w')
    file.write(josn_file)
    file.close()


def output_single_vof(images, file_name, save_dir):
    def save_im(fov, time):
        im_name = save_dir + '\\' + dic_name[fov] + '\\' + dic_name[fov] + '_' + file_suffix_ct[time] + '.tif'
        imsave(im_name, ims[time])
        return None
    ims_t = images.timepointcount
    ims_v = images.multipointcount
    times = images.metadata['CustomData|AcqTimesCache']
    ims_c = images.channels
    file_suffix_ct = [f'c{c}_t{t}' for c in range(ims_c) for t in range(ims_t)]
    dic_name = [file_name + f'_v{v}' for v in range(ims_v)]
    # for fov in range(ims_v):
    #     ims = np.array([images.image(fov, time)[..., 0] for time in tqdm(range(ims_t))])
    #     ims_y, ims_x = images.image(0, 0).shape[0:2]
    #     im_name = save_dir + '\\' + file_name + '_' + file_suffix[fov] + '.tiff'
    #     ims = ims.reshape(ims_t, 1, 1, ims_y, ims_x, 1)
    #     imsave(im_name, ims, imagej=True)
    #     print(f'file {im_name} saved!')

    get_ims = lambda fov, time: images.image(fov, time)[..., 0]
    # make directory for saving images
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    for fov in tqdm(range(ims_v)):  # iterate fov
        # ims = [images.image(fov, time)[..., 0] for time in range(ims_t)]
        ims = Parallel(n_jobs=-1, backend='threading')(delayed(get_ims)(fov, time) for time in range(ims_t))
        ims_y, ims_x = images.image(0, 0).shape[0:2]
        # os.mkdir(save_dir + '\\' + dic_name[fov])
        try:  # make sub folds
            os.mkdir(save_dir + '\\' + dic_name[fov])
        except FileExistsError:
            pass
        _ = Parallel(n_jobs=-1, backend='threading')(delayed(save_im)(fov, sub_time) for sub_time in range(ims_t))
        ###### -------------------------------------------------#########
    # out put the nd2 file parameters
    josn_file = {'file_name': file_name,
                 'time_point': times.tolist()}
    josn_file = json.dumps(josn_file)
    file = open(save_dir + '\\' + file_name + '_time.json', 'w')
    file.write(josn_file)
    file.close()


def output_vof_tstack2(images, file_name, save_dir):
    # define funcs
    get_ims = lambda fov, time: images.image(fov, time)[..., 0]

    def save_im(fov, ims):
        im_name = save_dir + '\\' + dic_name[fov] + '\\' + dic_name[fov] + '.tif'
        imsave(im_name, ims, imagej=True)
        return None

    ims_t = images.timepointcount
    ims_v = images.multipointcount
    times = images.metadata['CustomData|AcqTimesCache']
    ims_c = images.channels
    file_suffix_ct = [f'c{c}_t{t}' for c in range(ims_c) for t in range(ims_t)]
    dic_name = [file_name + f'_v{v}' for v in range(ims_v)]

    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    for fov in tqdm(range(ims_v)):
        ims = Parallel(n_jobs=-1, backend='threading')(delayed(get_ims)(fov, time) for time in range(ims_t))
        ims_y, ims_x = images.image(0, 0).shape[0:2]

        try:
            os.mkdir(save_dir + '\\' + dic_name[fov])
        except FileExistsError:
            pass

        ims = np.array(ims).astype(np.uint16)
        ims = ims.reshape(ims_t, 1, 1, ims_y, ims_x, 1)
        save_im(fov, ims)

    # out put the nd2 file parameters
    josn_file = {'file_name': file_name,
                 'time_point': times.tolist()}
    josn_file = json.dumps(josn_file)
    file = open(save_dir + '\\' + file_name + '_time.json', 'w')
    file.write(josn_file)
    file.close()


# %%
if __name__ =='__main__':
    save_dir = r'F:\ZJW_CP\20201118_NQ386_M5L6\single_fov'
    file_dir = r'Z:\panchu\image\mother machine\20201118_NQ386_M5L6\20201118_NQ386_M5L6_FRESH_RDM_PHASEL_001.nd2'
    file_name = file_dir.split('\\')[-1].split('.')[0]
    images = ND2MultiDim(file_dir)

    #%%
    nd2_ps = r'F:\MoMa_MZX\20211220.BW25113.MOPS-glycerol&RDM.micro\20211220.BW25113.MOPS-glycerol&RDM.micro.nd2'
    save_dir = r'F:\MoMa_MZX\20211220.BW25113.MOPS-glycerol&RDM.micro'
    nd2_imgs = ND2MultiDim(nd2_ps)
    output_single_vof(nd2_imgs, nd2_ps.split('\\')[-1].split('.')[0], save_dir)

    # %% out put vofs with stacked time points
    output_vof_tstack2(images, file_name, save_dir)
    # %% out-put vofs with single times points
    output_vof_tstack(images, file_name, save_dir)

