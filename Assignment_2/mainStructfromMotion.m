% This procedure runs all Structure from Motion stuff. 

% step 1 - initialise

consec_images_num = 4;      % 0, 3 or 4 - 
                            % 0 means that it taks the given pointviewmatrix  
                            
% step 3 - iterate (for each row)

if consec_images_num == 0
    disp('read given PVM..')
    PVM_given = readPVM();    % helper function
    
    % plotPVM(PVM_given)            % to give statistics and plot
    
    [M, S] = motionStructure(PVM_given);  % helper function
    
    size(S)
    PC = S;
else
    disp('chaining....')
    PVM = chaining();              % helper function
    
    % plotPVM(PVM)                    % to give statistics and plotting
    
    disp('dense blocking...')
    dense_blocks = find_dense_block(PVM, consec_images_num);  % helper function
    
    % amalgate all created S

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

function plotPVM(PVM, endColumn)
    if nargin == 1
        endColumn = size(PVM, 2)
    end
    % some statistics
    size(PVM)
    noPoints = sum(PVM(:)>0);
    noPVM = prod(size(PVM));
    percFilledPVM = noPoints/noPVM
    % plotting one set of rows
    figure
    X = PVM(1, :);
    Y = PVM(2, :);
    scatter(X, Y)
    % plotting all chained points
    count10 = 0;
    count20 = 0;
    count30 = 0;
    count50 = 0;
    count100 = 0;
    figure
    for ii = 1:endColumn
        % PVM_col = PVM(:, ii);
        PVM_col = PVM(PVM(:, ii)>0, ii);
        lenCol = size(PVM_col, 1);
        if lenCol > 0
            xIndex = [1:2:lenCol];
            yIndex = [2:2:lenCol];
            xPVM = PVM_col(xIndex);
            yPVM = PVM_col(yIndex);
            plot(xPVM, yPVM)
            hold on
        end
        if lenCol > 10
            count10 = count10+1;
        end
        if lenCol > 20
            count20 = count20+1;
        end
        if lenCol > 30
            count30 = count30+1;
        end
        if lenCol > 50
            count50 = count50+1;
        end
        if lenCol > 100
            count100 = count100+1;
        end
    end
    count10
    count20
    count30
    count50
    count100
end

% ########################
% |   Helper Functions   |
% ########################

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