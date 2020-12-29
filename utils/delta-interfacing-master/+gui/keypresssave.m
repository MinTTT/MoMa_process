function keypresssave(obj,evt,images,filenames,varargin)
% This function is to be attached to figures with the KeyPressFcn property,
% and it will save the "images" (cell of images) in the "filenames"
% location (cell of strings) if 'enter' is hit, or it will discard them if
% 'q' is hit. It can also write them to reject folders if you provide any.

ip = inputParser();
ip.addOptional('rejects',[]);
ip.parse(varargin{:});

switch evt.Key
    case 'return' % Save files:
        for ind1 = 1:numel(images)
            imwrite(images{ind1},filenames{ind1});
        end
        obj.UserData = 1;
    case 'q'
        if isempty(ip.Results.rejects)
            % Do nothing
        else
            for ind1 = 1:numel(images)
                imwrite(images{ind1},ip.Results.rejects{ind1});
            end
        end
        obj.UserData = 1;
end
        
        
        
