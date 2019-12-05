% clear;
% train = importdata('train.txt');
% fileID = fopen('part_train.txt', 'w');
% for i = 1:2:length(train)
%     name = train{i};
%     if contains(name,'_0.0.png')
%         if ~contains(name, 'SunCG') && ~contains(name, 'SceneNet')
%             fprintf(fileID, '%s\n', name);
%         end
%     end
% end
% fclose(fileID);
clear;
train = importdata('train.txt');
fileID = fopen('part_train.txt', 'w');
for i = 1:2:length(train)
    name = train{i};
    splits = split(name,' ');
    rgb = ['data/Realistic', splits{1}];
    if ~contains(name,'SunCG.png')
        if ~contains(name,'57_a74d64d3f8d94500816467e5d936db101')
            if exist(rgb, 'file')
                fprintf(fileID, '%s\n', name);
            end
        end
    end
end
fclose(fileID);