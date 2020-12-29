% This script can be used to generate a training set for cells tracking
% from the outputs of the segmentation U-Net.
% The user will have to click on cells in the new frame to generate the set
% See the supplementary information of our paper for more details.

% Parameters:
img_folder = '/home/jeanbaptiste/data/unet_trainingsets/segmentation_output/segmentation/img'; % original images folder
segall_folder = '/home/jeanbaptiste/data/unet_trainingsets/segmentation_output/segmentation/output'; % Segmentation output (from U-Net)
save_folder = '/home/jeanbaptiste/data/unet_trainingsets/training_set_2/tracking_set';
mkdir(fullfile(save_folder,'img'));
mkdir(fullfile(save_folder,'previmg'));
mkdir(fullfile(save_folder,'seg'));
mkdir(fullfile(save_folder,'segall'));
mkdir(fullfile(save_folder,'mother'));
mkdir(fullfile(save_folder,'daughter'));

% Potential samples:
pot_samples = dir(fullfile(segall_folder,'*.png'));
samplesleft = numel(pot_samples);

%% Run the "GUI":

numsamples = numel(dir(fullfile(save_folder,'seg','*.png')));

while(samplesleft)
    
    % Take one random sample out:
    r = randsample(samplesleft,1);
    samplesleft = setdiff(samplesleft,r);
    % Random sample -> frame_nb and files to load:
    filename = pot_samples(r).name;
    frame = str2num(filename((strfind(filename,'Frame')+length('Frame'))+(0:2))); % verify this
    if frame == 1 % Can't retrieve previous frame, just skipping this sample
        continue
    end
    prev_filename = sprintf('%s%03d%s', ...
                            filename(1:(strfind(filename,'Frame')+length('Frame')-1)), ...
                            frame-1, ...
                            filename((strfind(filename,'Frame')+length('Frame')+3):end));
    
    % Load images
    img = imread(fullfile(img_folder,filename));
    segall = imread(fullfile(segall_folder,filename));
    segall = imresize(segall,size(img),'nearest');
    previmg = imread(fullfile(img_folder,prev_filename));
    segall_prev = imread(fullfile(segall_folder,prev_filename));
    segall_prev = imresize(segall_prev,size(img),'nearest');
    
    [labeled, numcells] = bwlabel(segall_prev');
    labeled = labeled';
    
    labeled_new = bwlabel(segall',4)';
    
    h = figure(1);
    
    for cell = 1:numcells
        seg = labeled == cell;
        centroid = regionprops(seg,'Centroid');
            
        % Show input images:
        clf
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

        drawnow

        % Get the user to tap what cell is the mother and which on is
        % the mother:
        while(1)
            [x,y] = ginput(2);

            x = round(x);
            y = round(y);
            if ~isempty(x) && ~isempty(y) && any(x < 1 | x > size(segall,2) | y < 1 | y > size(segall,1))
                disp('Out of bounds, try again');
            elseif (numel(x) > 0 && segall(y(1),x(1)) == 0) || (numel(x) == 2 && segall(y(2),x(2)) == 0)
                disp('Hit outside of cells, try again');
            else
                break;
            end
        end

        % Select the cells from the user input:
        if numel(x) > 0
            mother = labeled_new==labeled_new(y(1),x(1));
        else
            mother = zeros(size(seg));
        end
        if numel(x) > 1
            daughter = labeled_new==labeled_new(y(2),x(2));
        else
            daughter = zeros(size(seg));
        end

        % Plot the result:
        subplot(1,6,5)
        imshow(mother);
        subplot(1,6,6)
        imshow(daughter);

        % Attach key press function: (this function saves the images if you hit
        % enter, and discards them if you hit q)
        h.UserData = 0;
        h.KeyPressFcn = ...
            {@gui.keypresssave, ... 
            {seg, previmg, img, segall, mother, daughter}, ...
            {fullfile(save_folder,'seg',sprintf('Sample%06d.png',numsamples+1)), ...
            fullfile(save_folder,'previmg',sprintf('Sample%06d.png',numsamples+1)), ...
            fullfile(save_folder,'img',sprintf('Sample%06d.png',numsamples+1)), ...
            fullfile(save_folder,'segall',sprintf('Sample%06d.png',numsamples+1)), ...
            fullfile(save_folder,'mother',sprintf('Sample%06d.png',numsamples+1)), ...
            fullfile(save_folder,'daughter',sprintf('Sample%06d.png',numsamples+1))}};
        waitfor(h,'UserData',1);
        numsamples = numel(dir(fullfile(save_folder,'img','*.png')));
        fprintf('%d samples\n',numsamples);
        
    end
end   
disp('Finished all potential samples!')
