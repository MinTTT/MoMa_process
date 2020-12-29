function IMG = lblsimg(L,varargin)

ip = inputParser();
ip.addParameter('Colors',[]);
ip.addParameter('Names',[]);
ip.addParameter('Text',true);
ip.parse(varargin{:})


lbls = unique(L(:));
lbls(lbls==0) = [];
if isempty(ip.Results.Names)
    labelnames = num2cell(lbls);
else
    labelnames = ip.Results.Names;
end

if isempty(ip.Results.Colors)
    IMG = label2rgb(L);
else
    colors = ip.Results.Colors;
    IMG = label2rgb(L,colors);
end

IMG = double(IMG)/255;

if(ip.Results.Text)
    r = regionprops(L,'Centroid');
    for ind1 = 1:numel(lbls)
        ll = lbls(ind1);
        name = labelnames{ind1};
        IMG = insertText(IMG,r(ll).Centroid,name,'AnchorPoint','Center','FontSize',8);
    end
end