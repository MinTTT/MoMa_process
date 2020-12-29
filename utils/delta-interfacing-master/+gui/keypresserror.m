function keypresserror(obj,evt)


switch evt.Key
    case '0'
        obj.UserData = 0;
    case 'q'
        obj.UserData = NaN;
    otherwise
        if isstrprop(evt.Key,'digit')
            obj.UserData = str2num(evt.Key);
        end
end
