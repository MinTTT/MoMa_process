% This script can be used to generate a training set for cells segmentation
% from a tif stack representing Ilastik output. Import Ilastik hdf5 outputs
% into fiji (with Ilastik plugin) and save as a tif stack.

% Parameters:
save_folder = '/home/jeanbaptiste/data/unet_trainingsets/training_set_3/segmentation_set';
mkdir(fullfile(save_folder,'img'));
mkdir(fullfile(save_folder,'seg'));
mkdir(fullfile(save_folder,'wei'));
inside_channel = 1;
membrane_channel = 2;

% Select original image file:
[transfile, transpath] = uigetfile('*.tif','Select original images stack');

% Select Ilastik output file:
[probsfile, probspath] = uigetfile('*.tif','Select Ilastik probabilities output');

%%
disp('Loading stacks into memory (this can take a long time)')
original = utilities.tifprocess(fullfile(transpath,transfile));
original = original{1};
probabilities = utilities.tifprocess(fullfile(probspath,probsfile),'NumberChannels',3);
probabilities = double(probabilities{1});

% lightly preprocess & watershed ilastik output:
for i = 1:size(probabilities,4)
    fprintf('[Ilastik output processing] Frame %d/%d\n',i,size(probabilities,4))
    inside = squeeze(probabilities(:,:,1,i))';
    membrane = squeeze(probabilities(:,:,2,i))';
    together = inside>.5 | membrane>.5;
    together = imfill(together,'holes');
    together = imopen(together,strel('square',1));
    together = bwareafilt(together,500);
    inside = inside.*double(together);
    membrane = membrane.*double(together);
    levels = ones(size(inside)).*Inf;
    levels(inside>.5) = 1;
    levels(membrane>.5) = 2;
    regions = watershed(levels);
    regions = (double(regions).*(levels<3))>0;
    regions = imfill(regions,'holes');
    % Contour smoothing:
    [L,N] = bwlabel(regions);
    final = zeros(size(regions));
    for l = 1:N
        dump = L==l;
        dump = imopen(dump,strel('square',3)); % smoothing contours
        dump = imclose(dump,strel('square',3)); % smoothing contours
        dump = imopen(dump,strel('square',3)); % smoothing contours
        final = final | dump;
    end
    regionslist(:,:,i) = final;
end


%% preprocess trans images:
[processed.rotation, ...
processed.chambers, ...
processed.XYdrift] = ...
preprocessing.MoMaPreprocessing(squeeze(original));

% Potential samples:
samplesleft = 1:(size(original,4)*size(processed.chambers,1));

%% Training samples already in the folder:
numsamples = numel(dir(fullfile(save_folder,'img','*.png')));

while(samplesleft)
    
    % Take one random sample out:
    r = randsample(samplesleft,1);
    samplesleft = setdiff(samplesleft,r);
    % random number -> frame & chamber
    frame = ceil(r/size(processed.chambers,1));
    chamber = mod(r,size(processed.chambers,1));
    chamber(chamber == 0) = size(processed.chambers,1);

    % Apply preprocessing:
    transimg = preprocessing.applypreprocessing(original,frame,processed, 'chamber', chamber, 'imadjust', true);
    segimg = preprocessing.applypreprocessing(regionslist,frame,processed, 'chamber', chamber);

    % Plot:
    h = figure(1);
    cla
    subplot(1,2,1)
    imshow(transimg);
    subplot(1,2,2)
    imshow(utilities.drawcontour(transimg,segimg));
    
    % Attach key press function: (this function saves the images if you hit
    % enter, and discards them if you hit q)
    h.UserData = 0;
    h.KeyPressFcn = ...
        {@gui.keypresssave, ... 
        {transimg, uint8(segimg)*255}, ...
        {fullfile(save_folder,'img',sprintf('Sample%06d.png',numsamples+1)), ...
        fullfile(save_folder,'seg',sprintf('Sample%06d.png',numsamples+1))}};
    waitfor(h,'UserData',1);
    numsamples = numel(dir(fullfile(save_folder,'img','*.png')));
    fprintf('%d samples\n',numsamples);

end
disp('Finished all potential samples!')

%% Weight maps: Run once the training set is complete. 
% (If you're going to merge training sets, preferrably run on assembled folder)

% Parameters
sigma = 2;
w0 = 12;
% The PNG mult fact is just to scale the weights to the dynamic range of
% 8-bit pngs. If you touch the parameters above, you might have to change
% PNGmultfact so that your data more or less fits the [0-255] range.
% IMPORTANT: Make sure that ALL samples in your dataset have the same PNG
% multiplication factor, otherwise you will assign higher weights to soe
% samples.
PNGmultfact = 25;
seg_list = dir(fullfile(save_folder,'seg','*.png'));


% Compute class frequency weights:
class0 = 0;
class1 = 0;

for segfile = 1:numel(seg_list)
    seg = imread(fullfile(seg_list(segfile).folder,seg_list(segfile).name));
    seg = seg>125;
    class0 = class0 + double(sum(~seg(:)));
    class1 = class1 + double(sum(seg(:)));
end

% Most represented class -> weight = 1. Counterweight the other
if class0 > class1
    weight_c0 = 1;
    weight_c1 = class0/class1;
else
    weight_c0 = class1/class0;
    weight_c1 = 1;
end

% Loop through seg files, compute distance array, and create weight map
% files:
for segfile = 1:numel(seg_list)
    
    %Read seg files:
    seg = imread(fullfile(seg_list(segfile).folder,seg_list(segfile).name));
    seg = seg>125;
    
    % Compute the distance mask:
    [L,N] = bwlabel(seg);
    dist_arr = Inf*ones([size(seg), max(N,2)]);
    for l = 1:N
        singlecell = L==l;
        dist_arr(:,:,l) = bwdist(singlecell);
    end
    dist_arr = sort(dist_arr,3);
    weight_dist = w0*exp((-(dist_arr(:,:,1) + dist_arr(:,:,2)).^2)/(2*sigma^2)); % Formula from Ronneberger et al.
    weight_dist = weight_dist.*(single(~seg));
    
    % Sum up all weights:
    weight_map = zeros(size(seg));
    weight_map(seg) = weight_c1;
    weight_map(~seg) = weight_c0;
    weight_map = weight_map + weight_dist;
    
    % Assign 0 weight to contours: (may or may not be a good idea)
    weight_map(xor(seg,imerode(seg,strel('disk',1)))) = 0;
    
    % Turn into an 8-bit PNG:
    weight_map = uint8(weight_map*PNGmultfact);
    imwrite(weight_map,fullfile(save_folder,'wei',seg_list(segfile).name));
    fprintf('%d/%d\n',segfile,numel(seg_list))
end