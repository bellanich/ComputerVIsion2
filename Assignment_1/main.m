%% Testing readPcd.m 
clear all
clc
close all

%{
BaseCloud = readPcd('0000000000.pcd');
TargetCloud = readPcd('0000000001.pcd');
%}

%% ICP

% Initilize 3 by 3 identity matrix

% Create function to search for all points in the base cloud to find the 
% closest point in the target cloud. 
% Will call this psi.
% Need to output a matrix of pairwise combinations (base point, nearest target point)

% THEN update approach, to get R and t using SVD.
% create if else statements to see when RMS changes

% just to try out some code and get used to the data

% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node

datapath = '../../data/'

s = load('source.mat')
s
size(s.source)

t = load('target.mat')
size(t.target)

a1 = readPcd([datapath, '0000000013_normal.pcd']);
a2 = readPcd([datapath, '0000000000.pcd']);

%% ICP - see below for main functions - I have to finilize ICP loop

[psi, psi_source] = det_matching(s.source', t.target');
[R, tr] = det_rotation(s.source', t.target', psi)
RMS = calc_RMS(s.source', t.target', psi)

%====================================================
%  helper function
%====================================================

function [R, t] = det_rotation(source, target, psi)
    % determine R and t with SVD
    % target is original set of points Nx3, psi give the corresponding 
    % indices of target-points which are closest to source
    N = length(psi)
    p = source(1:N, :)
    q = target(psi, :)
    % step1 - determine weighted cetroids of p and q
    % note: weighting w = 1 for all points
    pc = sum(p, 1)/N
    qc = sum(q, 1)/N;
    % step2 - centered vectors -> matrices D x N
    X = (p - pc)'
    Y = (q - qc)'
    % step3 - determine covariance matrix S = XWY'
    W = eye(N)    % all weights are = 1
    S = X*W*Y'
    % step 4 - singular value decomposition
    [U Sigma Vt] = svd(S)
    V = Vt'
    s = svd(S)
    Nr = size(Sigma, 1)
    rot = eye(Nr);
    rot(Nr, Nr) = det(V*U')
    R = V*rot*U'
    % Step 5 - optimal translation
    t = qc' - R*pc'
end

function [psi, psi_source] = det_matching(source, target)
    % source should be N x 3 matrix
    % target should be M x 3 matrix
    % psi is indices on target to map on source
    % N = size(source, 1)
    N = 5
    psi = [];
    psi_source = [];
    for ii = 1:N
        point = source(ii, :);
        [closest_index, closest_point] = find_closest_point(point, target);
        psi_source = cat(1, psi_source, closest_point);
        psi = cat(1, psi, closest_index);
    end
end

function [closest_index, closest_point] = find_closest_point(point, target)
    % point is 1 x 3
    % target is N x 3
    % closest_point is 1 x 3
    % used brute force to determine the closest_point in target which
    % is closest to point. 
    distances = pdist2(target, point);
    closest_index = find(distances == min(distances));
    closest_point = target(closest_index, :);
end

function [RMS] = calc_RMS (source, target, psi)
    % RMS distance calculated between source and psi(target)
    % target is original set of points, psi give the corresponding 
    % indices of target-points which are closest to source
    N = size(psi, 1);
    norms = sqrt(sum((source(1:N, :) - target(psi)).^2,2));
    RMS = sqrt(sum(norms, 1)/N);
end