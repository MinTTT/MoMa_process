% This script is used to compile lineages, stacks, and extract features
% such as fluorescence and cell length after DeLTA's segmentation.py and 
% tracking.py scripts have been run on preprocessed data (see
% preprocessing.m).
% We leave those scripts for reference, but we recommend using the
% pipeline.py script in DeLTA instead, as it is faster and over time we
% will stop supporting those legacy files.

%% Main parameters:
positions = [1]; % empty array [] for all positions
segfolder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/seg_output';
motherdaughterfolder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/track_output';
preprocfolder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/preprocessed';
postprocfolder = '/home/jeanbaptiste/data/delta_trainingsets/paulsson_dataset/postprocessed';
mkdir(postprocfolder)
fluo1chan = 1; % Fluorescence channel #1 (0 if not used)
fluo2chan = 0; % Fluorescence channel #2 (0 if not used)
fluo3chan = 0; % Fluorescence channel #3 (0 if not used)

if isempty(positions)
    % If empty, do all the positions that are in the preprocessing folder:
    dump = dir(fullfile(preprocfolder,'Position*.mat'));
    for ind1 = 1:numel(dump)
        positions(ind1) = str2num(dump(ind1).name(9:10));
    end
end


%% Refactor U-Net images into lineage data:

for position = positions
    clearvars res
    load(fullfile(preprocfolder,sprintf('Position%02d.mat',position)));
    
    for chamber = 1:size(proc.chambers,1)
        fprintf('Lineage reconstruction: Position %d, chamber %d\n',position,chamber);
        clearvars img seg segall mother daughter framenumbers
        samplenb = 0;

        % Recreating the stacks:
        for frame = 2:moviedimensions(4)
            currframename = sprintf('Position%02d_Chamber%02d_Frame%03d.png',position,chamber,frame);
            currframe = imread(fullfile(segfolder,currframename));
            prevframename = sprintf('Position%02d_Chamber%02d_Frame%03d.png',position,chamber,frame-1);
            prevframe = imread(fullfile(segfolder,prevframename));
            [lbls, numcells] = bwlabel(prevframe',4);
            lbls = lbls';
            for cellind = 1:numcells
                currcell = sprintf('Position%02d_Chamber%02d_Frame%03d_Cell%02d.png',position,chamber,frame,cellind);
                samplenb = samplenb + 1;
                segall(:,:,samplenb) = logical(currframe);
                seg(:,:,samplenb) = lbls==cellind;
                mother(:,:,samplenb) = imresize(imread(fullfile(motherdaughterfolder,['mother_',currcell])),size(squeeze(seg(:,:,1))));
                daughter(:,:,samplenb) = imresize(imread(fullfile(motherdaughterfolder,['daughter_',currcell])),size(squeeze(seg(:,:,1))));
                framenumbers(samplenb) = frame;
            end
        end

        % Process the stacks:
%         mother = mother > 125; % binarize
%         daughter = daughter > 125; % binarize
        if exist('segall')
            [lineage, labelsstack] = tracking.chamberlineage(segall, seg, mother, daughter, framenumbers);
        else % If chamber completely empty
            lineage = [];
            labelsstack = zeros([size(currframe),moviedimensions(4)]);
        end
        if size(labelsstack,3) < moviedimensions(4), labelsstack(:,:,(end+1):moviedimensions(4)) = 0; end % If all cells disappear form chamber at some point
        res(chamber).lineage = lineage;
        res(chamber).labelsstack = labelsstack;

    end
    save(fullfile(postprocfolder,sprintf('Position%02d.mat',position)),'res','-v7.3');
end


%% Create colored tif stacks for visualizing results:

chunksize = 50;

WriteNumber = false; % Write cell numbers on movie
WriteMother = false; % Write cell mother on movie (requires WriteNumber = true)

load(fullfile(preprocfolder,'Images.mat'));

for position = positions
    tic
    
    % Import processing:
    load(fullfile(preprocfolder,sprintf('Position%02d.mat',position)));
    load(fullfile(postprocfolder,sprintf('Position%02d.mat',position)));
    proc.chambers = round(proc.chambers);
    
%     proc.XYdrift = -proc.XYdrift;
    
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
    
        
    for c = 1:numel(chunks)
        
        fprintf('Processing chunk #%d/%d\n',c,numel(chunks));
        
        chunk = chunks{c};
    
        % Create blank variables:
        labelsmovieRGB = ones([moviedimensions([1,2]),3, numel(chunk)]);
        divsmovieRGB = ones([moviedimensions([1,2]),3, numel(chunk)]);



        % Run through chambers:
        for chamber = 1:size(proc.chambers,1)

            % Initialize:
            clearvars labelsstack labelsRGB divsRGB
            lineage = res(chamber).lineage;
            colors = colors_c{chamber};
            fprintf('Writing stacks: Position %d, chamber %d\n',position,chamber);

            % Resize labelsstack to original size:
            chamberpos = proc.chambers(chamber,:);
            dump = res(chamber).labelsstack;
            for i = 1:size(dump,3), labelsstack(:,:,i) = imresize(squeeze(dump(:,:,i)),chamberpos(4:-1:3),'nearest'); end


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
        cropbox = [proc.chambers(1,1), ...
                    proc.chambers(1,2), ...
                    (proc.chambers(end,1)-proc.chambers(1,1)+proc.chambers(1,3)), ...
                    proc.chambers(1,4)];
        seg = squeeze(mean(labelsmovieRGB,3)<1); % dirty but works


        clearvars tiledmovie  
        for chunk_el = 1:numel(chunk)

            frame = chunk(chunk_el);
            fprintf('Tiling frame #%d\n',frame)
            img = preprocessing.applypreprocessing(squeeze(images{position}(:,:,1,:)),frame,proc);
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
        utilities.writestack(fullfile(postprocfolder,sprintf('Position%02d_groundtruth.tif',position)),tiledmovie);
    end
        toc
    
end



%% Extract fluorescence and other features:

disp('Loading images, this can take a long time...')
load(fullfile(preprocfolder,'Images.mat'));

for position = positions
    fprintf('Starting position %d\n',position)
    
    % Import processing:
    load(fullfile(preprocfolder,sprintf('Position%02d.mat',position)));
    load(fullfile(postprocfolder,sprintf('Position%02d.mat',position)));

    for chamber = 1:size(proc.chambers,1)
        
        % Initialize:
        tic
        clearvars labelsstack
        lineage = res(chamber).lineage;
        chamberpos = proc.chambers(chamber,:);
        fprintf('Features extraction: Position %d, chamber %d\n',position,chamber);
        
        % Resize labelsstack to original size:
        dump = res(chamber).labelsstack;
        for i = 1:size(dump,3), labelsstack(:,:,i) = imresize(squeeze(dump(:,:,i)),chamberpos(4:-1:3),'nearest'); end
        res(chamber).labelsstack_resized = labelsstack;


        % Pasting labelstack into stack of same dims as original movie:
        labelssingle = zeros(moviedimensions([1,2,4]));
        y_indexes = chamberpos(2)+ (0:(chamberpos(4)-1));
        x_indexes = chamberpos(1)+ (0:(chamberpos(3)-1));
        labelssingle( ...
            y_indexes(y_indexes>0&y_indexes<=moviedimensions(1)), ...
            x_indexes(x_indexes>0&x_indexes<=moviedimensions(2)), ...
            : ...
        ) = res(chamber).labelsstack_resized(   y_indexes>0&y_indexes<=moviedimensions(1), ...
                                                x_indexes>0&x_indexes<=moviedimensions(2),:);

        % Running through frames:
        for frame = 1:moviedimensions(4)

            % Get fluo images:
            if fluo1chan
                fluo1img = preprocessing.applypreprocessing(squeeze(images{position}(:,:,fluo1chan,:)),frame,proc);
            end
            if fluo2chan
                fluo2img = preprocessing.applypreprocessing(squeeze(images{position}(:,:,fluo2chan,:)),frame,proc);
            end
            if fluo3chan
                fluo3img = preprocessing.applypreprocessing(squeeze(images{position}(:,:,fluo3chan,:)),frame,proc);
            end
            
            % Get cell numbers:
            cf = squeeze(labelssingle(:,:,frame));
            cells = unique(cf(:));
            cells(cells==0) = [];
            
            % Extract features for each cell:
            if ~isempty(cells) % In case of empty chamber
                for cell = cells'
                    
                    % Get index position of current frame number in cell's
                    % lineage
                    framei = find(lineage(cell).framenbs==frame);
                    % Get mask for single cell:
                    mask = cf==cell;
                    
                    % Extract fluo:
                    if fluo1chan
                        fluo1 = mean(fluo1img(mask),'all');
                        lineage(cell).fluo1(framei) = fluo1;
                    end
                    if fluo2chan
                        fluo2 = mean(fluo2img(mask),'all');
                        lineage(cell).fluo2(framei) = fluo2;
                    end
                    if fluo3chan
                        fluo3 = mean(fluo3img(mask),'all');
                        lineage(cell).fluo3(framei) = fluo3;
                    end
                    
                    % Extract morphological properties:
                    props = regionprops(mask,'Area','MajorAxisLength','MinorAxisLength');
                    lineage(cell).area(framei) = props(1).Area;
                    lineage(cell).length(framei) = props(1).MajorAxisLength;
                    lineage(cell).width(framei) = props(1).MinorAxisLength;
                    
                end
            end
        end
        res(chamber).lineage = lineage;
        save(fullfile(postprocfolder,sprintf('Position%02d.mat',position)),'res','-v7.3');
        toc
    end
end
