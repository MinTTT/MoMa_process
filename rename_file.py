#%%
import os
#file_lis=[]
#for f in os.scandir(path = r'E:\Pan\毕设\cytometry'):
#    file_lis.append(f.name)
#%%
path = r'G:\ZJW_CP\20200805_moma_pecj35M5_RDM_Glu\s_fov\xy25'
file_lis = [f.name for f in os.scandir(path = path) if f.is_file()]

for file in file_lis:
    suffix = file.split('.')[-1]
    prefix = file.split('.')[0]
    if suffix == 'tiff':
        new_na = prefix + '.tif'
        os.rename(path + '\\' + file , path + '\\' + new_na  )
