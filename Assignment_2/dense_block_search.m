clear all 
clc
close all

%  === Generate random toy point-view matrix for testing purposes ======
W = 15;
L = 15;
test_matrix = randi([-30,100],W,L);
test_matrix(find(test_matrix < 0)) = -1;
test_matrix
 
%%
% ============== Parameters to pass to function ===================
% number of consec images
consec_images_num = 2;
densest_blocks = find_dense_block(test_matrix, consec_images_num);

% to loop through dense blocks
num_densest_blocks = length(cellfun('size', densest_blocks, 2));

% currently, print out each dense block saved in the list
for block = 1:num_densest_blocks
    % gives you a single 
    densest_blocks{block}
end


% ============= Sliding dense block  window search =================
function  dense_blocks = find_dense_block(point_view_file, consec_images_num)

    [row_num, col_num] = size(point_view_file);
    
    search_width = consec_images_num*2;

    % list of dense matrices found during sliding wind search
    dense_blocks = cell(row_num - search_width, 1);

    % For every sliding window, find and save dense block
    for i = 0: (row_num - search_width)
        data_window = point_view_file(1 + i : search_width + i, :);

        % use all nonzero columns to make condensed block
        dense_cols = find(all(data_window ~= -1)); 
        condense_wind = data_window(:, dense_cols);

        % save condensed block in list 
        dense_blocks{i+1} = condense_wind;
    end
    
    % ----- Uncomment to find retrieve the densest block ----
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
