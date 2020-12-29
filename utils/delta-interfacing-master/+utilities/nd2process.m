function imagelist = nd2process(varargin)
% This function loads an .nd2 file and then outputs it as files list. You
% can specify the filename or not (in the second case a file selection browser will
% pop up) and you can specify a list of positions to extract as well as a
% list of channels to extract.
% You will need to install the bioformats toolbox for this function to
% work: https://docs.openmicroscopy.org/bio-formats/5.7.2/users/matlab/index.html


% Process inputs:
ip = inputParser();
ip.addOptional('input','',@ischar);
ip.addParameter('Positions',[],@isnumeric);
ip.addParameter('Channels',[],@isnumeric);
ip.parse(varargin{:});

if isempty(ip.Results.input)
    [f,p] = uigetfile('.nd2','Select ND2 file');
    nd2file = fullfile(p,f);
else
    nd2file = ip.Results.input;
end
data = bfopen(nd2file);

if isempty(ip.Results.Positions)
    positions = 1:size(data,1);
else
    positions = ip.Results.Positions;
end



% Extract the data:
for indpos = 1:numel(positions)
    position = positions(indpos);
    datapos = data{position,1};
    for frame = 1:size(datapos,1)
        [channel, timepoint] = getinfo(datapos{frame,2});
        stack(:,:,channel,timepoint) = datapos{frame,1};
    end
    
    if isempty(ip.Results.Channels)
        % Nothing to do
    else
        stack = stack(:,:,ip.Results.Channels,:);
    end
    
    imagelist{indpos} = stack;
    clearvars stack;
end
        
      
function [channel, timepoint] = getinfo(infostr)
ch = strfind(infostr,'C?=');
slash = strfind(infostr(ch:end),'/');
channel = str2num(infostr((ch+3):(ch+slash(1)-2)));

tp = strfind(infostr,'T?=');
slash = strfind(infostr(tp:end),'/');
timepoint = str2num(infostr((tp+3):(tp+slash(1)-2)));