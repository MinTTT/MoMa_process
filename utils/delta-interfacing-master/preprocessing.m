% This script is used to pre-process microscopy movies, crop out images
% of the chambers, and save them to disk to then be used by DeLTA's
% segmentation.py and tracking.py files. 
% We leave those scripts for reference, but we recommend using the
% pipeline.py script in DeLTA instead, as it is faster and over time we
% will stop supporting those legacy files.

%% Main parameters:
tiffile = 'C:\DeepLearning\DeLTA_data\Nadia_movies\20190925\20190925_reca_dinb_sula_cm_cropped.tif';
positions = []; % empty array [] for all positions
imgfolder = 'C:\DeepLearning\DeLTA_data\Nadia_movies\preprocessed\img';
mkdir(imgfolder) % making sure the folder does exist
preprocfolder = 'C:\DeepLearning\DeLTA_data\Nadia_movies\preprocessed\';
mkdir(preprocfolder) % making sure the folder does exist

%% get data from tif file:
images = utilities.tifprocess(tiffile, ...
    'NumberPositions',19, ...
    'NumberChannels', 2, ...
    'KeepPositions',positions);
save(fullfile(preprocfolder,'Images.mat'),'images','-v7.3');

%% Preprocess images, extract single chambers, and write to disk:
drift_limits = [-20 -20; 20 20]; % This is by how much the drift can change *PER TIME STEP*

for position = 1:numel(images)
    
    moviedimensions = size(images{position});
    trans = squeeze(images{position}(:,:,1,:)); % Here we assume the trans channels is #1

    [proc.rotation, ...
    proc.chambers, ...
    proc.XYdrift] = ...
    preprocessing.MoMaPreprocessing(trans,'driftrange',drift_limits,'driftrange_adapt',true,'chambersspacing',90:.1:100);

    
    save(fullfile(preprocfolder,sprintf('Position%02d.mat',position)),'tiffile','proc','moviedimensions','-v7.3');
    
    for frame = 1:size(trans,3)
        for chamber = 1:size(proc.chambers,1)
            transimg = preprocessing.applypreprocessing(trans,frame,proc, 'chamber', chamber, 'imadjust', true);
            filename = sprintf('Position%02d_Chamber%02d_Frame%03d.png',position,chamber,frame);
            imwrite(transimg,fullfile(imgfolder,filename));
            disp(filename)
    end
end