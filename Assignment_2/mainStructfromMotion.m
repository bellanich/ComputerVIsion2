% This procedure runs all Structure from Motion stuff. 

% step 1 - initialise

consec_images_num = 3;      % 0, 3 or 4 - 
                            % 0 means that it taks the given pointviewmatrix  

% step 3 - iterate (for each row)

if consec_images_num == 0
    disp('read given PVM..')
    PVM_given = readPVM;
    
    [M, S] = motionStructure(PVM_given);
    PC = S;
else
    disp('chaining....')
    PVM = chaining();
    
    disp('dense blocking...')
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

function plot_3D(S)
    figure
    X = S(1, :);
    Y = S(2, :);
    Z = S(3, :);
    scatter3(X, Y, Z, 3, 'r' )
end

function P = readPVM

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