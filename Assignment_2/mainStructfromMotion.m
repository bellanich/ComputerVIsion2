% This procedure runs all Structure from Motion stuff. 

% step 1 - initialise

consec_images_num = 3;      % 0, 3 or 4 - 
                            % 0 means that it taks the given pointviewmatrix  
                            
% step 3 - iterate (for each row)

if consec_images_num == 0
    disp('read given PVM..')
    PVM_given = readPVM();    % helper function
    
    [M, S] = motionStructure(PVM_given);  % helper function
    
    PC = S;
else
    disp('chaining....')
    PVM = chaining();              % helper function
    
    disp('dense blocking...')
    dense_blocks = find_dense_block(PVM, consec_images_num);  % helper function
    % step 4 - amalgate all created S

    % to loop through dense blocks
    num_densest_blocks = length(cellfun('size', dense_blocks, 2));

    % currently, print out each dense block saved in the list

    for block = 1:num_densest_blocks
        % gives you a single 
        A = dense_blocks{block};
        [M, S] = motionStructure(A);
        if block == 1
            S0 = S;
            PC = S;
        else
            [d, Z] = procrustes(S0, S);
            PC = cat(2, PC, Z);
        end
    end

    sizePC = size(PC)
    % clean up for outliers
    remove_inx1 = find(PC(3, :) > 1);
    remove_inx2 = find(PC(3, :) < -1);
    remove_inx = [remove_inx1, remove_inx2];
    
    PC(:, remove_inx) = [];
    sizePC = size(PC)

end

plot_3D(PC)

% ========================================================
% helper function
% ========================================================

function [M, S] = motionStructure(Dsparse, base)
% function delivers M and S matrices 

if nargin == 1
    base = 0;
end

% step 0 - sample dense block - assumption that all is dense 
% look for points which are in 90% (parameter) of projections
threshold = 0.9;
D = [];
[noProjections, noPoints] = size(Dsparse);
for ii = 1:noPoints
    points = Dsparse(:, ii);
    if sum(points(points(1) > 0)) / noProjections > threshold
        D = cat(2, D, points);
    end
end

% step: process whole D (if base == 0) or walk through D with steps = base
[noProjections, noPoints] = size(D);
if base == 0
    [M, S] = calcMS(D);
else
    step = 2*base;
    noIterations = floor(noPoints/step);
    for ii = 1:noIterations
        Diter = D((ii-1)*step+1:ii*step, :);
        [Miter, Siter] = calcMS(Diter);
        if ii == 1
            Sstart = Siter;
            S = Sstart;
            sizeStart = size(S)
        else
            [~,Z] = procrustes(Sstart,Siter);
            S = cat(2, S, Z);
        end
    end
    sizeS = size(S)
    M = 0
end
end

% ===============================================================
% helper function
% ===============================================================

function [M, S] = calcMS(D)
% D is dense D 
    % step 1 - point normalisation
    mD = mean(D, 2);
    sizemD=size(mD);
    Dnorm = D-mD;
    % step 2 - SVD - and reduction to rank 3
    [U, W, V] = svd(Dnorm);
    U3 = U(:, 1:3);
    W3 = W(1:3, 1:3);
    V3 = V(:, 1:3) ;        
    D3 = U3 * W3 * V3';
    % step 3 - construct M for Motion and S for Structure
    M = U3 * W3^(1/2);
    S = W3^(1/2) * V3';
end



function PVM = chaining()

    % load images
    images = load_images();


    % initialize padded PVM
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

    % add rows
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

    PVM = remove_padding(PVM);

end

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
    indices = any(PVM > -1); 
    PVM = PVM(:, indices);
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

function  dense_blocks = find_dense_block(point_view_matrix, consec_images_num)

    %{
        ============= Sliding dense block  window search =================
        :params:
        point_view_matrix (PVM): matrix built from chaining.m
        consec_images_num: number of consecutive images to search for dense
        blocks in

        :returns:
        dense_blocks: list of all created dense blocks from PVM
    %}

    % calculate needed parameters 
    [row_num, col_num] = size(point_view_matrix);
    search_width = consec_images_num*2;
    search_times = floor((row_num - search_width)/2);

    % initialize list to save dense matrices found during search
    cell_length = ceil(search_times/2);
    dense_blocks = cell(cell_length, 1);

    % For every sliding window, build and save a dense block
    for i = 0: 2: search_times
        
        data_window = point_view_matrix(1 + i : search_width + i, :);

        % use all nonzero columns to make condensed block
        dense_cols = find(all(data_window ~= -1)); 
        condense_wind = data_window(:, dense_cols);

        % save condensed block in list 
        block_ind = (i/2) + 1;
        dense_blocks{block_ind} = condense_wind;
        
    end
    
    % ---- Uncomment to find retrieve the biggest densest block ----
    % densest_block = max_search(dense_blocks);
end

% ==== helper function to search for biggest dense block =========
function max_found = max_search(my_cell)

    % get list of sizes of the matrixes stored in cell
    my_cell_sizes = cellfun('size', my_cell, 2);
    
    % find largest sized matrix
    max_size = max(my_cell_sizes);
    argmax = find(max_size == my_cell_sizes);
    % return
    max_found = my_cell{argmax};
end

function plot_3D(S)
    figure
    X = S(1, :);
    Y = S(2, :);
    Z = S(3, :);
    scatter3(X, Y, Z, 3, 'r' )
end

function P = readPVM()

% PoineViewMatrix.txt has 2X101 lines (views) with each 215 3D points
fileID = fopen('PointViewMatrix.txt');
tmp = textscan(fileID, '%f');
fclose(fileID);

% create P 2M x N array
% pointview: 2M X N, M: 101 viewdoc s, N: 215 3D points
P = zeros(202, 215);
for ii = 1:202
    for jj = 1:215
        index = jj + 215*(ii-1);
        tmp{1}(index);
        P(ii, jj) = tmp{1}(index);
    end
end

end