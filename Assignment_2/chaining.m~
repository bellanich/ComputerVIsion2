clear
clc
close all

%% load images
images = load_images();


%% initialize padded PVM
PVM = -1 .* ones(2*size(images, 3), 4000);
threshold = 4;

image1 = images(:,:,1);
image2 = images(:,:,2);

[matches, features1, features2] = keypoint_matching( ...
                                image1, image2, threshold);

first_row = remove_doubles([features1(1:2, matches(1, :)); features2(1:2, matches(2, :))]);

for i = 1:size(first_row, 2)
    for k = 1:4
        PVM(k, i) = first_row(k, i);
    end
end

max_index = i;


%% add rows to PVM
for i = 2:size(images, 3)-1
    
    i_current = 2*i;
    i_next = 2*(i+1);

    [matches, features1, features2] = keypoint_matching( ...
                                  images(:,:,i), images(:,:,i+1), threshold);

    next_row = remove_doubles([features1(1:2, matches(1, :)); features2(1:2, matches(2, :))]);

    buffer = [];

    for j = 1:size(next_row, 2)

        index = find_index(PVM(i_current-1:i_current, :), next_row(1:2, j));

        if index < 1
            buffer = [buffer, next_row(3:4, j)];
        else
            PVM(i_next-1:i_next, index) = next_row(3:4, j);
        end
    end

    for k = 1:size(buffer, 2)
        PVM(i_next-1:i_next, max_index+k) = buffer(1:2, k);
    end

    max_index = max_index + size(buffer, 2);
end

%%
% ########################
% |   Helper Functions   |
% ########################
function images = load_images
images = [];
directory = pwd + "/Data/House/";
    for i = 1:49
        filename = sprintf("frame000000%02d.png", i);
        current_image = imread(strcat(directory, filename));
        images = cat(3, images, im2single(current_image));
    end
end

function PVM = remove_padding(PVM)
end

function index = find_index(array1, coords)
    
    index = 0;

    for i = 1:size(array1, 2)
        if coords(1) == array1(1, i) && coords(2) == array1(2, i)
            index = i;
            break;
        end
    end
end

function row = remove_doubles(array)
    row = unique(array', 'rows', 'stable')';
end

