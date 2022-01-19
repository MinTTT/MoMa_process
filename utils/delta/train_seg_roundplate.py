'''
This script trains the chambers segmentation U-Net.

@author: jblugagne
'''
from utils.delta.model import unet_chambers
from utils.delta.data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Files:
DeLTA_data = r'F:\PHA_library'
training_set = os.path.join(DeLTA_data, 'train_set')
model_file = os.path.join(DeLTA_data, 'models', 'Unet_plate.hdf5')

# Parameters:

target_size = (512, 512)
input_size = target_size + (1,)
batch_size = 1
epochs = 100
steps_per_epoch = 250

# Data generator:
data_gen_args = dict(
    rotation=3,
    shiftX=.1,
    shiftY=.1,
    zoom=.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotations_90d=True,
    histogram_voodoo=True,
    illumination_voodoo=True
)

myGene = trainGenerator_seg(batch_size,
                            os.path.join(training_set, 'plate_train'),
                            os.path.join(training_set, 'plate_mask'),
                            None,
                            augment_params=data_gen_args,
                            target_size=target_size)

# Define model_for_colony:
model = unet_chambers(input_size=input_size)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)

# Train it:
model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint])
