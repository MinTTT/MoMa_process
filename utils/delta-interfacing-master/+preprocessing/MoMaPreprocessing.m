function [rotation, chambers, XYdrift] = MoMaPreprocessing(images,varargin)

    
    % Parameters:
    ip = inputParser();
    ip.addParameter('thetarange',-10:0.1:10);
    ip.addParameter('display',0);
    ip.addParameter('chambersspacing',35:.1:45);
    ip.addParameter('chamberswidth',20);
    ip.addParameter('driftrefpos',[+10 45]);
    ip.addParameter('rotation',[]);
    ip.addParameter('chambers',[]);
    ip.addParameter('driftrange',[]);
    ip.addParameter('driftrange_adapt',false);
    ip.parse(varargin{:});
    
    display = ip.Results.display;
    thetarange = ip.Results.thetarange; % Range of angles to consider for correction
    chambersspacing = ip.Results.chambersspacing; % Range of possible chambers spacings to consider
    chamberswidth = ip.Results.chamberswidth; % Width of the cropping rectangle for chamber images
    driftrefpos = ip.Results.driftrefpos; % Y pos relative to end of chambers boxes + height
    rotation = ip.Results.rotation;
    chambers = ip.Results.chambers;
    driftrange = ip.Results.driftrange;
    driftrange_adapt = ip.Results.driftrange_adapt;
    
    I0 = squeeze(images(:,:,1)); % Pull first image for reference
    nbimages = size(images,3);

    if ~isempty(chambers) && ~isempty(rotation) % If user provides rotation and chambers positions, only run the XY drift routine
        for ind1 = 1:size(chambers,1)
            chamberscenters(ind1,1) = chambers(ind1,2) + (chambers(ind1,4)-20)/2 + 20;
            chamberscenters(ind1,2) = chambers(ind1,1) + chambers(ind1,3)/2;
        end
        spacing = mean(diff(chamberscenters(:,2)));
        
    else
        % Correct angle:
        rotation = anglecorr(I0,thetarange);
        rotI = imrotate(I0,rotation,'bilinear','crop');
        if display
            figure('Name','Angle correction');
            imagesc(rotI)
        end

        % Find chambers in image:
        [chambers,chamberscenters,spacing] = getChamberboxes(rotI, chambersspacing, chamberswidth);

        if display
            figure('Name','Chambers ID');
            imagesc(rotI);
            hold on
            for ind1 = 1:size(chambers,1)
                rectangle('Position',chambers(ind1,:),'EdgeColor','y');
            end
            hold off
        end
    end
    
    % Compute drift correction:
    referencebox = [chambers(1,1) + chambers(1,3)/2 ... X pos
        chambers(1,2) + driftrefpos(1) - driftrefpos(2) ... Y pos
        chambers(end,1) - chambers(1,1) ... Width
        driftrefpos(2) ... Height
        ];
    
    if display
        figure('Name','Drift correction')
    end
    
    drift_search = driftrange;
    for ind1 = 2:nbimages
        I = imrotate(squeeze(images(:,:,ind1)),rotation,'bilinear','crop');
        [XYdrift(ind1-1,1), XYdrift(ind1-1,2)] = computedriftcorr(I, rotI, chambers, drift_search);
        
        if driftrange_adapt % Update the drift range search box:
            drift_search(:,1) = driftrange(:,1) + [XYdrift(ind1-1,1); XYdrift(ind1-1,1)];
            drift_search(:,2) = driftrange(:,2) + [XYdrift(ind1-1,2); XYdrift(ind1-1,2)];
        end
        
        fprintf('[Drift correction] %d/%d: x=%d - y=%d pixels\n',ind1-1,nbimages,XYdrift(ind1-1,1),XYdrift(ind1-1,2))
        if display
            imagesc(imtranslate(I,-XYdrift(ind1-1,:)));
            for ind2 = 1:size(chambers,1)
                rectangle('Position',chambers(ind2,:),'EdgeColor','y');
            end
            drawnow
        end
    end
    if ~exist('XYdrift')
        XYdrift = [];
    end
end

function [thetacorr] = anglecorr(I,thetarange)

    % Compute the linear hough transform and extract the best lines:
    [H,theta,rho] = hough(edge(I),'Theta',thetarange);
    P = houghpeaks(H,20,'threshold',ceil(0.3*max(H(:))));
    lines = houghlines(edge(I),theta,rho,P,'FillGap',10,'MinLength',20);
    fprintf('[Angle correction] Found %d reference lines\n',numel(lines));
    thetacorr = mean([lines.theta]);
    fprintf('[Angle correction] Estimated correction angle: %.2f�\n',thetacorr);
end

function [chambers,chamberscenters,spacing] = getChamberboxes(rotI, chambersspacing, chamberswidth)

    % "Parameters"
    proto = imread('proto.tif');
%     proto = imresize(proto,2);
    rectsize = [chamberswidth,size(proto,1)];
    
    % Compute normalized xcorr on the template chamber:
    C = normxcorr2(double(proto),rotI);
    Cc = imcrop(C,[fliplr(ceil(size(proto)./2)) fliplr(size(rotI))]);

    % Find the aligned peaks:
    [chamberscenters,spacing] = findchambers(Cc,chambersspacing);

    % Compute bounding boxes for each chamber:
    for ind1 = 1:size(chamberscenters,1)
        center = chamberscenters(ind1,:);
        chambers(ind1,:) = [center(2) - rectsize(1)/2, center(1) - rectsize(2)/2 - 20, rectsize + [0 20]];
    end
end

function [xcorr, ycorr] = computedriftcorr(I0, I1, chambers, driftrange)

    cutmargin = 30; % To avoid the black region from the imrotate output
    refIMG = imgradient(I0(cutmargin:chambers(1,2),cutmargin:(end-cutmargin)));
    compIMG = imgradient(I1(cutmargin:chambers(1,2),cutmargin:(end-cutmargin)));
    
    refIMG(refIMG<150) = 0;
    compIMG(compIMG<150) = 0;
    
    C = normxcorr2(refIMG,compIMG);
    if ~isempty(driftrange)
        C = imcrop(C,[flip(size(C))/2 + driftrange(1,:) diff(driftrange,1,1)]);
    end

    [ypeak, xpeak] = find(C==max(C(:)));
    ycorr = ypeak-(size(C,1)+1)/2;
    xcorr = xpeak-(size(C,2)+1)/2;
    
    if ~isempty(driftrange)
        ycorr = ycorr + (driftrange(1,2) + driftrange(2,2))/2;
        xcorr = xcorr + (driftrange(1,1) + driftrange(2,1))/2;
    end      

end

%% Utilities

function [locations,spacingout] = findchambers(I,spacingrange)

    maxres = 0;
    params = [];
    spacingout = 0;
    for ind1 = 1:size(I,1)
        [linemax,linepeaks,spacing] = scanthroughline(spacingrange,I(ind1,:));
        if linemax > maxres
            maxres = linemax;
            ypos = find(linepeaks);
            locations = [ind1*ones(size(ypos)),ypos];
            spacingout = spacing;
        end
    end
    fprintf('[Chambers identification] Found %d chambers on line %d\n',size(locations,1),locations(1,1));

end    

function [maxval,params,spacingout] = scanthroughline(spacingrange,line)

    maxval = 0;
    params = [];
    spacingout = 0;
    for spacing = spacingrange
        for shift = 1:spacing
            beads = genline(shift,spacing,numel(line));
            if sum(beads.*line') > maxval
                maxval = sum(beads.*line');
                params = beads;
                spacingout = spacing;
            end
        end
    end
end

function [beads] = genline(shift,spacing,length)
    beads = zeros(length,1);
    beads(round(shift:spacing:end)) = 1;
end