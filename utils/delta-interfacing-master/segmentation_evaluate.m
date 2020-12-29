% This script can be used to evaluate/curate segmentation results after 
% running DeLTA's segmentation.py script.
% The curated samples can then be mixed with an existing training set.

img_folder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/img'; % original images folder
segall_folder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/seg_output'; % Segmentation output (from U-Net)

h = figure(1);

samples = dir(fullfile(segall_folder,'*.png'));
samplesleft = numel(samples);

numprocessed = 0;
errors = 0;

checkedfiles = {};
rejectedfiles = {};
errorsnb = [];

%%

while samplesleft
    
    % Take one random sample out: (randomized to reduce bias)
    r = randsample(samplesleft,1);
    samplesleft = setdiff(samplesleft,r);
    filename = samples(r).name;
    
    % Load images
    img = imread(fullfile(img_folder,filename));
    seg = imread(fullfile(segall_folder,filename));
    img = imresize(img,size(seg));
    
    % Plot:
    h = figure(1);
    cla
    subplot(1,2,1)
    imshow(img);
    subplot(1,2,2)
    imshow(utilities.drawcontour(img,seg>0));
    
    % Attach key press function: (this function saves the images if you hit
    % enter, and discards them if you hit q)
    h.UserData = -1;
    h.KeyPressFcn = @gui.keypresserror;
    waitfor(h,'UserData');
    if ~isnan(h.UserData)
        errors = errors + h.UserData;
        checkedfiles{end+1} = filename;
        errorsnb(end+1) = errors;
        
        [~, dump] = bwlabel(seg>0);
        numprocessed = numprocessed + dump;
    else
        rejectedfiles{end+1} = filename;
    end
    
    save('errorrate.mat','checkedfiles','errorsnb','rejectedfiles');
    
    fprintf('%.2f%% error rate\n',100*errors/numprocessed);
    
end