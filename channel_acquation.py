#%%
from skimage import io
import numpy as np
import utils.rotation as rot
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import utils.registration as reg

def crop_images(im, ang):
    im = rot.apply_rotate_and_cleanup(im, ang)[0]
    return im

def rotate_fov(ims):
    num_fov = len(ims)
    rot_angles = Parallel(n_jobs=-1)(delayed(rot.find_rotation)(ims[idx_fov, ...]) for idx_fov in range(num_fov))
    rot_ims = Parallel(n_jobs=-1)(delayed(crop_images)(ims[idx_fov, ...], rot_angles[idx_fov]) for idx_fov in range(num_fov))
    return np.array(rot_ims), np.array(rot_angles)


#%%
ims_dir = r'E:\20200729_moma_pecj35M5_RDM_Glu\all_fig'
ims_prefix = r'20200730_0001002_c0_v0_t'

times_list = [0, 1, 2, 3]


#%%
ims = np.array([io.imread(ims_dir + '\\' + ims_prefix + str(time) + '.tiff') for time in times_list])
# ims shape: (lenoftimes_list, y, x)
rotated_ims, rot_angles = rotate_fov(ims)

#%%
def reg_img(ref_im, im):
    chi = reg.translation_2x1d(ref_im, im)
    reged_im = reg.shift_image(im, chi[0])
    return reged_im

reged_ims = np.array(Parallel(n_jobs=-1)(delayed(reg_img)(rotated_ims[0, ...], rotated_ims[time, ...]) for time in times_list))

#%%
fig1, ax = plt.subplots()
io.imshow(np.vstack([reged_ims[i, ...] for i in range(3)]))
fig1.show()

