function imagelist = tifprocess(varargin)
% This function loads an tif file and then outputs it as a cell of image arrays. You
% can specify the filename or not (in the second case a file selection browser will
% pop up) and you can specify a list of positions to extract as well as a
% list of channels to extract.


% Process inputs:
ip = inputParser();
ip.addOptional('input','',@ischar);
ip.addParameter('KeepPositions',[],@isnumeric);
ip.addParameter('NumberChannels',1,@isnumeric);
ip.addParameter('NumberPositions',1,@isnumeric);
ip.addParameter('KeepChannels',[],@isnumeric);
ip.addParameter('dtype','uint16');
ip.parse(varargin{:});

if isempty(ip.Results.input)
    [f,p] = uigetfile('.tif','Select TIF file');
    tiffile = fullfile(p,f);
else
    tiffile = ip.Results.input;
end

nbpositions = ip.Results.NumberPositions;
if ~isempty(ip.Results.KeepPositions)
    positions = ip.Results.KeepPositions;
else
    positions = 1:nbpositions;
end

nbchannels = ip.Results.NumberChannels;
if ~isempty(ip.Results.KeepChannels)
    channels = ip.Results.KeepChannels;
else
    channels = 1:nbchannels;
end

if isfolder(tiffile) % Image sequence folder
    % Here we're going to assume images are ordered in the pos > chan >
    % frame order
    imlist = dir(fullfile(tiffile,'*.tif'));
    nbframes = numel(imlist)/(nbpositions*nbchannels);
    
    for indpos = 1:numel(positions)
        position = positions(indpos);
        for indchan = 1:numel(channels)
            channel = channels(indchan);
            for frame = 1:nbframes
%                 index = ...
%                     (frame-1) * nbpositions * nbchannels + ...
%                     (position-1) * nbchannels + ...
%                     channel;
                index = (position-1) * nbframes * nbchannels + ...
                    (channel-1) * nbframes + ...
                    frame;
                stack(:,:,indchan,frame) = imread(fullfile(tiffile,imlist(index).name));
            end
        end
        imagelist{indpos} = stack;
    end
    
else % Tiff file
    stackinfo = imfinfo(tiffile);
    nbframes = numel(stackinfo)/(nbpositions*nbchannels);
    if floor(nbframes)~=nbframes
        error('Number of positions and channels provided is invalid'); % Unless the nb frame is different between each pos
    end

    % First read the data as it is in the file:
    mImage=stackinfo(1).Width;
    nImage=stackinfo(1).Height;
    stack_unordered=zeros(nImage,mImage,numel(stackinfo),ip.Results.dtype); % Allocate once
    TifLink = Tiff(tiffile, 'r');
    for i=1:numel(stackinfo)
       TifLink.setDirectory(i);
       stack_unordered(:,:,i) = TifLink.read();
    end
    TifLink.close();
    
    % Extract the data:
    for indpos = 1:numel(positions)
        position = positions(indpos);
        for indchan = 1:numel(channels)
            channel = channels(indchan);
            for frame = 1:nbframes
                index = ...
                    (frame-1) * nbpositions * nbchannels + ...
                    (position-1) * nbchannels + ...
                    channel;
                stack(:,:,indchan,frame) = stack_unordered(:,:,index);
            end
        end
        imagelist{indpos} = stack;
    end
end