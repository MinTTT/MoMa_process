'''
This script trains the cell segmentation U-Net

@author: jblugagne
'''
from model import unet_seg
from data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint

# Files:
DeLTA_data = 'C:/DeepLearning/DeLTA_data/'
training_set = DeLTA_data + 'mother_machine/training/segmentation_set/train_multisets/'
model_file = DeLTA_data + 'mother_machine/models/unet_moma_seg_multisets.hdf5'

# Parameters:
target_size = (256, 32)
input_size = target_size + (1,)
batch_size = 10
epochs = 200
steps_per_epoch = 250

#Data generator:
data_gen_args = dict(
                    rotation = 0.5,
                    shiftX=.05,
                    shiftY=.05,
                    zoom=.15,
                    horizontal_flip=True,
                    histogram_voodoo=True,
                    illumination_voodoo=True)

myGene = trainGenerator_seg(batch_size,
                           training_set + 'img/',
                           training_set + 'seg/',
                           training_set + 'wei/',
                           augment_params = data_gen_args,
                           target_size = target_size)


# Define model:
model = unet_seg(input_size = input_size)
model.summary()
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)


# Train it:
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])