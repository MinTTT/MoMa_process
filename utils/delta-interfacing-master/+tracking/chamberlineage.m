function [lineage, labelsstack] = chamberlineage(segmentation, seed_cells, mother_cells, daughter_cells, framenumbers)

% Initialize variables:
lineage = repmat(struct('pixels',{},'trackingscore',[],'framenbs',[],'mother',[],'daughters',[]),1,0);
imdim = size(segmentation);
labelsstack = zeros([imdim(1:2) max(framenumbers)]);
% Set first frame stack values:
dump = zeros(imdim(1:2));
for i = find(framenumbers==2)
    dump = dump + squeeze(seed_cells(:,:,i));
end
[labelsstack(:,:,1), N] = topdownbwlabel(dump,4);
% Initialize lineage:
for i = 1:N
    cell_img = labelsstack(:,:,1) == i;
    lineage = lineageupdate(lineage,i,cell_img,1,1,0,0);
end


%%%% Loop through the frames:
for framenb = sort(unique(framenumbers))
    indexes = find(framenumbers==framenb);
    segall = squeeze(segmentation(:,:,indexes(1)));
    [labeled,N] = topdownbwlabel(segall,4);
    labeled_prev = zeros(size(segall));
    scores = zeros(numel(indexes),N,2);
    labelsstack(:,:,framenb) = 0;
    labelstack_prevframe = squeeze(labelsstack(:,:,framenb-1));
        
    
    
    %%%%% Compute scores:
    for cell = 1:numel(indexes)
        i = indexes(cell);
        seg = squeeze(seed_cells(:,:,i));
        mother = double(squeeze(mother_cells(:,:,i)))/255;
        daughter = double(squeeze(daughter_cells(:,:,i)))/255;
        
        labeled_prev = labeled_prev + double(seg)*cell;
        
        combined_mother = labeled.*double(mother>0);
        combined_daughter = labeled.*double(daughter>0);
        
        for hit = setdiff(unique(combined_mother(:)),0)'
            if hit
                scores(cell,hit,1) = sum(mother(combined_mother==hit),'all')./sum(labeled==hit,'all');
            end
        end
        for hit = setdiff(unique(combined_daughter(:)),0)'
            if hit
                scores(cell,hit,2) = sum(daughter(combined_daughter==hit),'all')./sum(labeled==hit,'all');
            end
        end
    end
    
    
    
    %%%%% Curate attribution matrix
    scores(scores<.3) = 0; % Filter out small scores
    
    attrib_matrix = zeros(size(scores));
    % Detect conflicts:
    for hit = 1:size(scores,2)
        scores(:,hit,:) = scores(:,hit,:)/sum(scores(:,hit,:),'all'); % Normalize scores
        if sum(scores(:,hit,:),'all') > 0
            
            [score_mo,winner_mo] = max(scores(:,hit,1));
            [score_dau,winner_dau] = max(scores(:,hit,2));
            if score_mo >= score_dau
                attrib_matrix(winner_mo,hit,1) = 1;
            else
                attrib_matrix(winner_dau,hit,2) = 1;
            end
        end
    end
    
    for cell = 1:size(scores,1)
        if sum(attrib_matrix(cell,:,1)) > 1
            candidates = find(attrib_matrix(cell,:,1));
            attrib_matrix(cell,candidates(2:end),1) = 0; % Just keep the first one
        end
        if sum(attrib_matrix(cell,:,2)) > 1
            candidates = find(attrib_matrix(cell,:,2));
            attrib_matrix(cell,candidates(2:end),2) = 0; % Just keep the first one
        end
    end




    %%%%% Update Lineage, update labelstack
    for cell = 1:size(attrib_matrix,1)
        
        % Cell number in the previous labeled frame:
        labelstack_nb = unique(labelstack_prevframe(labeled_prev==cell));
        
        % Get mother/daughter attributions (if any)
        mother = find(attrib_matrix(cell,:,1),1,'first');
        daughter = find(attrib_matrix(cell,:,2),1,'first');
        mother_img = zeros(size(labeled));
        daughter_img = zeros(size(labeled));
        
        if mother
            mother_img = labeled==mother;
            if daughter
                daughter_img = labeled==daughter;
                % Create daughter:
                lineage = lineageupdate(lineage,numel(lineage)+1,daughter_img,scores(cell,daughter,2),framenb,labelstack_nb,0);
                % Update mother:
                lineage = lineageupdate(lineage,labelstack_nb,mother_img,scores(cell,mother,1),framenb,0,numel(lineage));
            else
                lineage = lineageupdate(lineage,labelstack_nb,mother_img,scores(cell,mother,1),framenb,0,0);
            end
        else
            if daughter % Shouldn't happen but we'll process it anyways
                daughter_img = labeled==daughter;
                % Create daughter:
                lineage = lineageupdate(lineage,numel(lineage)+1,daughter_img,scores(cell,daughter,2),framenb,labelstack_nb,0);
            end
        end
        
        % Update labels stack:
        if mother
            labelsstack(:,:,framenb) =  labelsstack(:,:,framenb) + ...
                                        double(mother_img).*labelstack_nb;
        end
        if daughter                     
            labelsstack(:,:,framenb) =  labelsstack(:,:,framenb) + ...
                                        double(daughter_img).*numel(lineage);
        end
    end
    
    
    
    %%%%% Look for orphans, update accordingly:
    for hit = 1:size(attrib_matrix,2)
        if sum(attrib_matrix(:,hit,:),'all') == 0
            orphan_img = labeled==hit;
            lineage = lineageupdate(lineage, ...
                        numel(lineage)+1, ... New cell
                        orphan_img, ... Cell image
                        NaN, ... % No tracking score
                        framenb, ...
                        0, ... No mother
                        0); % No daughter
                    
            labelsstack(:,:,framenb) =  labelsstack(:,:,framenb) + ...
                                    double(orphan_img).*numel(lineage);
        end
    end
    
end


function [L,N] = topdownbwlabel(I,conn)
if size(I,1) > size(I,2) % Image is probably upright
    [L,N] = bwlabel(I',conn);
    L = L';
else
    [L,N] = bwlabel(I,conn);
end


function lineage = lineageupdate(lineage, cellid, cellimg, score, framenb, mothernb, daughternb)

if cellid == 0 % Typically a glitch appearing out of nowhere, discard
elseif cellid <= numel(lineage)
    % If lineage already updated for this framenb, create this cell as a
    % daughter:
    lineage(cellid).pixels{end+1} = find(cellimg(:));
    lineage(cellid).trackingscore(end+1) = score;
    lineage(cellid).framenbs(end+1) = framenb;
    lineage(cellid).daughters(end+1) = daughternb;
else
    lineage(cellid).pixels = {find(cellimg(:))};
    lineage(cellid).trackingscore = score;
    lineage(cellid).framenbs = framenb;
    lineage(cellid).mothernb = mothernb;
    lineage(cellid).daughters = daughternb;
end