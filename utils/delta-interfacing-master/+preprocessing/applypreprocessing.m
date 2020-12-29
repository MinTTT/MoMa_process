function [img] = applypreprocessing(il,ind1,params, varargin)

ip = inputParser();
ip.addParameter('chamber',0);
ip.addParameter('imadjust',0);
ip.parse(varargin{:})


img = squeeze(il(:,:,ind1));
if ip.Results.imadjust
    img = imadjust(img);
end

img = imrotate(img,params.rotation,'bilinear','crop');

if ind1 > 1
    img = imtranslate(img,params.XYdrift(ind1-1,:));
end

if ip.Results.chamber>0
    img = imcrop(img,params.chambers(ip.Results.chamber,:));
end