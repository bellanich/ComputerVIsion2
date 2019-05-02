clear all 
clc
close all

%  === Generate random toy point-view matrix for testing purposes ======
W = 22;
L = 22;
test_matrix = randi([-30,100],W,L);
test_matrix(find(test_matrix < 0)) = -1;
test_matrix
 
%%
% ============== Parameters to pass to function ===================
% number of consec images
consec_images_num = 2;
densest_blocks = find_dense_block1(test_matrix, consec_images_num)

% to loop through dense blocks
% num_densest_blocks = length(cellfun('size', densest_blocks, 2));
% 
% % currently, print out each dense block saved in the list
% for block = 1:num_densest_blocks
%     % gives you a single 
%     densest_blocks{block}
% end


% ============= Sliding dense block  window search =================
function  dense_blocks = find_dense_block1(point_view_file, consec_images_num)

    % calculate needed parameters 
    [row_num, col_num] = size(point_view_file);
    search_width = consec_images_num*2;
    search_times = floor((row_num - search_width)/2);

    % initialize list to save dense matrices found during search
    cell_length = ceil(search_times/2);
    dense_blocks = cell(cell_length, 1);

    % For every sliding window, build and save a dense block
    for i = 0: 2: search_times
        
        data_window = point_view_file(1 + i : search_width + i, :);

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
