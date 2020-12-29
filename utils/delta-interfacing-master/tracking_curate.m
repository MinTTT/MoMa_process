% This script can be used to curate tracking results after running DeLTA's
% tracking.py script. The curated samples can then be mixed with an
% existing training set

% Parameters:
old_folder = '/home/jeanbaptiste/data/unet_trainingsets/training_set_1/tracking_set/train'; % original images folder
new_folder = '/home/jeanbaptiste/data/unet_trainingsets/training_set_1/tracking_set_curated_dump'; % Segmentation output (from U-Net)
reject_folder = '/home/jeanbaptiste/data/unet_trainingsets/training_set_1/tracking_set_rejected_dump';

mkdir(fullfile(new_folder,'img'));
mkdir(fullfile(new_folder,'previmg'));
mkdir(fullfile(new_folder,'seg'));
mkdir(fullfile(new_folder,'segall'));
mkdir(fullfile(new_folder,'mother'));
mkdir(fullfile(new_folder,'daughter'));

mkdir(fullfile(reject_folder,'img'));
mkdir(fullfile(reject_folder,'previmg'));
mkdir(fullfile(reject_folder,'seg'));
mkdir(fullfile(reject_folder,'segall'));
mkdir(fullfile(reject_folder,'mother'));
mkdir(fullfile(reject_folder,'daughter'));

% Samples:
samples = dir(fullfile(fullfile(old_folder,'img'),'*.png'));
samplesleft = numel(samples);

%% Run the "GUI":

h = figure(1);

while(samplesleft)
    
    % Take one random sample out: (randomized to reduce bias)
    r = randsample(samplesleft,1);
    samplesleft = setdiff(samplesleft,r);
    filename = samples(r).name;
    
    % Load images
    img = imread(fullfile(old_folder,'img',filename));
    previmg = imread(fullfile(old_folder,'previmg',filename));
    seg = imread(fullfile(old_folder,'seg',filename));
    segall = imread(fullfile(old_folder,'segall',filename));
    mother = imread(fullfile(old_folder,'mother',filename));
    daughter = imread(fullfile(old_folder,'daughter',filename));
    
    
    centroid = regionprops(seg,'Centroid');
    
    % Show images:
    clf(h)
    subplot(1,6,1)
    imshow(seg);
    subplot(1,6,2)
    imshow(previmg);
    hold on
    plot(centroid(1).Centroid(1),centroid(1).Centroid(2),' .r','MarkerSize',30);
    subplot(1,6,3)
    imshow(img);
    subplot(1,6,4)
    imshow(segall);
    subplot(1,6,5)
    imshow(mother);
    subplot(1,6,6)
    imshow(daughter);
    
    % Attach key press function: (this function saves the images to the new
    % folder if you hit enter, and to the reject folder if you hit q)
    h.UserData = 0;
    h.KeyPressFcn = ...
        {@gui.keypresssave, ... 
        {seg, previmg, img, segall, mother, daughter}, ...
        {fullfile(new_folder,'seg',filename) ...
        fullfile(new_folder,'previmg',filename) ...
        fullfile(new_folder,'img',filename) ...
        fullfile(new_folder,'segall',filename) ...
        fullfile(new_folder,'mother',filename) ...
        fullfile(new_folder,'daughter',filename)}, ...
        {fullfile(reject_folder,'seg',filename) ...
        fullfile(reject_folder,'previmg',filename) ...
        fullfile(reject_folder,'img',filename) ...
        fullfile(reject_folder,'segall',filename) ...
        fullfile(reject_folder,'mother',filename) ...
        fullfile(reject_folder,'daughter',filename)}};
    waitfor(h,'UserData',1);

end