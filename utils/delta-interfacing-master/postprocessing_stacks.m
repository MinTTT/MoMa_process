% This script will generate tif stacks to visualize the results of the 
% DeLTA pipeline.py script
% It is based on postprocess.m and as such requires the data to be
% reformatted to the 'legacy' data format with legacy_res().
% (No need to do anything about it here, legacy_res is called at the 
% beginning of the script.)
% 
% The 4 quadrants in the stack frames are (top-left, clockwise):
% 1 - Original chambers region
% 2 - Segmentation result
% 3 - Tracking result
% 4 - Divisions (red: mother, blue: daughter)

% DeLTA pipeline results folder:
postprocfolder = '/home/jeanbaptiste/data/DeLTA_data/mother_machine/evaluation/delta_results';
% Images sequence folder:
tifseq = '/home/jeanbaptiste/data/DeLTA_data/mother_machine/evaluation/sequence';
positions = []; % empty array [] for all positions

if isempty(positions)
    % If empty, do all the positions that are in the preprocessing folder:
    dump = dir(fullfile(postprocfolder,'Position*.mat'));
    for ind1 = 1:numel(dump)
        posnb = regexp(dump(ind1).name,'\d*','Match');
        positions(ind1) = str2num(posnb{1});
    end
end

% Get images:
images = utilities.tifprocess(tifseq, ...
    'NumberPositions',15, ... Number of positions in images sequence folder
    'NumberChannels', 2, ... Number of channels
    'KeepPositions',positions);

% Misc:
WriteNumber = false; % Write cell numbers on movie
WriteMother = false; % Write cell mother on movie (requires WriteNumber = true)
chunksize = 50; % For memory reasons, only process images 50 by 50.

% Run through positions:
for p = 1:numel(positions)
    position = positions(p);
    tic
    
    % Import processing:
    load(fullfile(postprocfolder,sprintf('Position%06d.mat',position)));  
    res = legacy_res(res);  
    proc.chambers = round(proc.chambers);
    
    % Generate colors for each chamber:
    clearvars colors_c
    for chamber = 1:size(proc.chambers,1)
        lineage = res(chamber).lineage;
        colors_c{chamber} = hsv(numel(lineage));
        colors_c{chamber} = colors_c{chamber}(randperm(size(colors_c{chamber},1)),:);
    end
    
    % Subdivide the stack in manageable chunks that will be appended
    % iteratively to the tiff file:
    c_ind = 0;
    chunks = {};
    while c_ind < moviedimensions(4)
        chunks{end+1} = (c_ind+1):min(c_ind+chunksize,moviedimensions(4));
        c_ind = c_ind+chunksize;
    end
    
    % Process by chunks:
    for c = 1:numel(chunks)
        
        fprintf('Processing chunk #%d/%d\n',c,numel(chunks));
        chunk = chunks{c};
    
        % Create blank variables:
        labelsmovieRGB = ones([moviedimensions([1,2]),3, numel(chunk)]);
        divsmovieRGB = ones([moviedimensions([1,2]), 3, numel(chunk)]);

        % Run through chambers:
        for chamber = 1:size(proc.chambers,1)

            % Initialize:
            clearvars labelsstack labelsRGB divsRGB
            lineage = res(chamber).lineage;
            colors = colors_c{chamber};
            fprintf('Writing stacks: Position %d, chamber %d\n',position,chamber);

            % Resize labelsstack to original size:
            chamberpos = proc.chambers(chamber,:);
            labelsstack = res(chamber).labelsstack_resized;

            % Run through frames:
            for chunk_el = 1:numel(chunk)

                frame = chunk(chunk_el);
                % LABELS
                % Get labels and labelled frame:
                L = labelsstack(:,:,frame);
                lbls = unique(L(:));
                lbls(lbls==0) = [];

                % Create image:
                LRT = tracking.lblsimg(L,'Colors',colors,'Text',WriteNumber);

                % Concatenate:
                labelsRGB(:,:,:,chunk_el) = LRT;

                % DIVISIONS
                % Run through labels, see if division at current timepoint:
                D = zeros(size(L));
                colorsdivs = [];
                dump = 0;
                if ~isempty(lbls)
                    for l = lbls'
                        daughter = lineage(l).daughters(lineage(l).framenbs==frame);
                        if daughter
                            dump = dump+1;
                            D(L==l) = dump;
                            dump = dump+1;
                            D(L==daughter) = dump;
                            colorsdivs = cat(1,colorsdivs,[1 0 0; 0 0 1]);
                        end
                    end
                end

                % Create image:
                DRT = tracking.lblsimg(D,'Colors',colorsdivs,'Text',WriteNumber);

                % Concatenate:
                divsRGB(:,:,:,chunk_el) = DRT;

            end

            % Paste into chamber position:
            y_indexes = chamberpos(2)+ (0:(chamberpos(4)-1));
            x_indexes = chamberpos(1)+ (0:(chamberpos(3)-1));
            labelsmovieRGB( ...
                y_indexes(y_indexes>0&y_indexes<=moviedimensions(1)), ...
                x_indexes(x_indexes>0&x_indexes<=moviedimensions(2)), ...
                :, ...
                : ...
            ) = labelsRGB(  y_indexes>0&y_indexes<=moviedimensions(1), ...
                            x_indexes>0&x_indexes<=moviedimensions(2),:,:);

            divsmovieRGB( ...
                y_indexes(y_indexes>0&y_indexes<=moviedimensions(1)), ...
                x_indexes(x_indexes>0&x_indexes<=moviedimensions(2)), ...
                :, ...
                : ...
            ) = divsRGB(    y_indexes>0&y_indexes<=moviedimensions(1), ...
                            x_indexes>0&x_indexes<=moviedimensions(2),:,:);
        end

        % Crop box:
        cropbox = [min(proc.chambers(:,1)), ...
                    min(proc.chambers(:,2)), ...
                    (proc.chambers(end,1)-proc.chambers(1,1)+proc.chambers(1,3)), ...
                    max(proc.chambers(:,2)+proc.chambers(:,4))-min(proc.chambers(:,2))];
        seg = squeeze(mean(labelsmovieRGB,3)<1); % dirty but works


        % Tile quadrants
        clearvars tiledmovie  
        for chunk_el = 1:numel(chunk)

            frame = chunk(chunk_el);
            fprintf('Tiling frame #%d\n',frame)
            img = preprocessing.applypreprocessing(squeeze(images{p}(:,:,1,:)),frame,proc);
            img_crop = imcrop(img,cropbox);
            seg_crop = imcrop(squeeze(seg(:,:,chunk_el)),cropbox);

            cont = utilities.drawcontour(img_crop,seg_crop);
            img_crop = utilities.grs2rgb(img_crop,gray());

            labels_crop = imcrop(squeeze(labelsmovieRGB(:,:,:,chunk_el)),cropbox);
            divs_crop = imcrop(squeeze(divsmovieRGB(:,:,:,chunk_el)),cropbox);

            tiled = imtile({img_crop,cont,labels_crop,divs_crop},'GridSize',[2 2],'BorderSize',[1 1]);
            if ~exist('tiledmovie')
                %First frame, allocate memory:
                tiledmovie = zeros([size(tiled), numel(chunk)]);
            end

            tiledmovie(:,:,:,chunk_el) = tiled;
        end

        % Write stack to disk: (if file already exists, it will append to
        % it
        utilities.writestack(fullfile(postprocfolder,sprintf('Position%06d.tif',position)),tiledmovie);
    end
    toc
    
end
