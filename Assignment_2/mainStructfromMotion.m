% This procedure runs all Structure from Motion stuff. 

% step 1 - initialise

% step 2 - create point-view matrix (Chaining)

disp('chaining....')
PVM = chaining();
%%
% step 3 - iterate (for each row)

disp('dense blocking...')

consec_images_num = 4;   % or 4
dense_blocks = find_dense_block(PVM, consec_images_num);

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

remove_inx = find(PC(PC(3, :) > 1));
PC(:, remove_inx) = [];

sizePC = size(PC)

plot_3D(PC)

% ========================================================
% helper function
% ========================================================

function plot_3D(S)
    figure
    X = S(1, :);
    Y = S(2, :);
    Z = S(3, :);
    scatter3(X, Y, Z, 3, 'r' )
end
