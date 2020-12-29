function res = legacy_res(res)

res = cell2mat(res);
for chamber = 1:numel(res)
    res(chamber).lineage = cell2mat(res(chamber).lineage);
    res(chamber).labelsstack = double(permute(res(chamber).labelsstack,[2,3,1]));
    res(chamber).labelsstack_resized = double(permute(res(chamber).labelsstack_resized,[2,3,1]));
end
