clear all
clc
close all

%MERGE merge all point clouds into each other
%   The point clouds will be merged into each other two different
%   protocols. The first computes the ratations and translations of each
%   frame 

% Set parameters for ICP
selectionType = 3;      % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                        
nr_samples = 500;        % only used for selectionType = 2 or 3
maxIterations = 100;
diffRMS = 0.0005;       % convergence if small improvement in RMS
sampling_rate = 1;

% load point clouds and create a selection
point_clouds = load_point_clouds();
selection = create_selection(sampling_rate);

%% 
% initialize lists with rotation and translation matrices
Rm = [];
tm = [];

% Get rotation and translation matrices for point clouds.
for i = 1:size(selection, 2)
    
    Htarget = point_clouds(:,:,selection(1, i));
    Hsource = point_clouds(:,:,selection(2, i));
    
    % remove all z > 1;
    Htarget = Htarget(Htarget(:, 3)<1, :);
    Hsource = Hsource(Hsource(:, 3)<1, :);
    
    % strip of 4th colom to be used ICP
    Htarget3 = Htarget(:, 1:3);
    Hsource3 = Hsource(:, 1:3);

    
    [RMS, message, R, t] = ICP(Hsource3, Htarget3, selectionType, ...
                            nr_samples, maxIterations, diffRMS);
    
    Rm = cat(3, Rm, R);
    tm = cat(2, tm, t);   
end

%%
first_cloud = point_clouds(:,:,1);
merged_cloud = first_cloud(first_cloud(:, 3) < 1, :);
plotCloud(merged_cloud, '1st cloud')
%%
for j = 1:size(selection, 2)
    
    %  R is 3x3 and t is 3x1. A column and row added for being able to rotate
    %  the N x 4 point clouds, while keepint the 4th column
    rotation = eye(3);
    translation = [0;0;0];
    for k = 1:j
        rotation = rotation * Rm(:,:,k);
        translation = translation + tm(:,k);
    end
    
    RCol = cat(2, cat(1, rotation, [0, 0, 0]), [0; 0; 0; 1]);
    tCol = cat(1, translation, 0);

    next_cloud = point_clouds(:,:,selection(2, j));
    next_cloud = next_cloud(next_cloud(:, 3)<1, :);
    rotated_cloud = [];
    for k = 1:size(next_cloud, 1)
        next_point = next_cloud(k, :)';
        rotated_cloud = cat(1, rotated_cloud, (RCol * next_point + tCol)');
    end
    merged_cloud = [merged_cloud; rotated_cloud]; 
end
plotCloud(merged_cloud, 'Merged')
%%
% ================
% Helper functions
% ================
function point_clouds = load_point_clouds()
% load point clouds in an array
    datapath =  './Data/';
    point_clouds = [];

    for i = 1:99
        % load point cloud
        filename = [datapath, '00000000', num2str(i, '%02i'), '.pcd'];
        point_cloud = readPcd(filename);

        % create padding and concatenate
        point_cloud(size(point_cloud, 1) : 100000, :) = 2;
        point_clouds = cat(3, point_clouds, point_cloud);
    end
end

function selection = create_selection(step_size)
% genarate selection of indices with a given step size that can be used to
% select the right point clouds
    selection = [];
    for i = 1 : step_size: 99-step_size
        selection = cat(2, selection, [i, i+step_size]');
    end
end


function plotCloud (pointCloud, figure_title)
    % pointCLoud should be N X 4 matrix
    figure
    X = pointCloud(:, 1);
    Y = pointCloud(:, 2);
    Z = pointCloud(:, 3);
    C = pointCloud(:, 4);
    fscatter3(X, Y, Z, C);
    title = figure_title;
end

