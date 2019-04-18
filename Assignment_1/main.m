% Testing readPcd.m and reading source and target data

clear all
clc
close all

% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode'); 
% please note...    ./ is current node
datapath = './Data/';


Htarget = readPcd([datapath, '0000000000.pcd']);
Hsource = readPcd([datapath, '0000000010.pcd']);
% remove all z > 1;
Htarget = Htarget(Htarget(:, 3)<1, :);
Hsource = Hsource(Hsource(:, 3)<1, :);

% plot these
plotCloud(Htarget, 'target')
plotCloud(Hsource, 'source')

% strip of 4th colom to be used ICP
Htarget3 = Htarget(:, 1:3);
Hsource3 = Hsource(:, 1:3);

% ICP method is tested

selectionType = 3;      % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                        % 4 is NOT implemented yet
                        
nr_samples = 500;        % only used for selectionType = 2 or 3
maxIterations = 100;
diffRMS = 0.0005;       % convergence if small improvement in RMS

[RMS, message, R, t] = ICP(Hsource3, Htarget3, selectionType, nr_samples, maxIterations, diffRMS)

%  R is 3x3 and t is 3x1. A column and row added for being able to rotate
%  the N x 4 point clouds, while keepint the 4th column
RCol = cat(2, cat(1, R, [0, 0, 0]), [0; 0; 0; 1]);
tCol = cat(1, t, 0);

%  here we rotate the source and plot them - these fit quite nice the
%  target (original)
HsourceRotated = (RCol * Hsource' + tCol)';
plotCloud(HsourceRotated, 'rotated')

% ==================================================================================
% helper function
% ==================================================================================

function plotCloud (pointCloud, title)
    % pointCLoud should be N X 4 matrix
    figure
    X = pointCloud(:, 1);
    Y = pointCloud(:, 2);
    Z = pointCloud(:, 3);
    C = pointCloud(:, 4);
    fscatter3(X, Y, Z, C);
    title = title;
end
