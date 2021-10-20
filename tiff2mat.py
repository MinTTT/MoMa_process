#%%
import tifffile as tiff
from scipy.io import savemat
import numpy as np
import h5py
import os


#%%

dir = r'/media/fulab/Fulab-hcy-data/20211012rp-cheZ30'
tifs = [tif for tif in os.listdir(dir) if tif.split('.')[-1] == 'tiff']

#%%
tif = tiff.imread(os.path.join(dir, tifs[0]))



#%%
sub_img = tif[0:10, ...]
h5_im = h5py.File(os.path.join(dir, tifs[0] + '.h5'), 'w')
h5_im.create_dataset('image_data', data=np.asfortranarray(sub_img))  # note matlab use column major order
h5_im.create_dataset('axes', data='TYX')
h5_im.close()


#%%
# bin_tif = h5py.File(os.path.join(dir, tifs[0] + '.bin'), 'w')
data_dict = {f'{i}': tif[i, ...].astype(int) for i in range(len(tif))}
mat_tif = savemat(os.path.join(dir, tifs[0] + '.mat'), data_dict,
                  long_field_names=True)
# bin_tif.create_dataset('data', data=tif)
# bin_tif.close()