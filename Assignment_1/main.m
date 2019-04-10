%% Testing readPcd.m and reading source and target data

clear all
clc
close all

% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node
datapath = '../../data/';

source = load('source.mat');
size(source.source);
target = load('target.mat');
size(target.target);

%{
BaseCloud = readPcd([datapath, '0000000013_normal.pcd']);
TargetCloud = readPcd([datapath, '0000000000.pcd']);
%}

%% ICP method is tested on source.mat and target.mat

selectionType = 3       % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                        % 4 is NOT implemented yet
                        
nr_samples = 25         % only used for selectionType = 2 or 3
maxIterations = 100;
diffRMS = 0.0005;       % convergence if small improvement in RMS

[RMS, message, Rot, trans] = detICP(source.source', target.target', selectionType, nr_samples, maxIterations, diffRMS)

%====================================================
%  helper function
%====================================================

function [RMS, message, R, t] = detICP(source, target, selectionType,  nr_samples, maxIterations, diffRMS)
   
    % Initilize R, tr, RMS
    dim = 3;
    R = eye(dim);
    t = zeros(dim, 1);
    sourceRotated = source;    % init: no rotation or translation
    
    % if selectionType == 2, then initialize sample indices
    if selectionType == 2
            disp('selectionType=2 - fixed nr of samples')
            Nmax = size(source, 1);
            N = nr_samples;   
            sampleInd = randi(Nmax, N, 1);
    else
            sampleInd = [];
    end
    
    % start the loop
    ii = 1;
    oldRMS = 1000000000;
    
    % psiT is the target reordered in such a way that points correspond with points in source 
    [~, targetPsi, ~, ~] = det_matching(source, target);    
    RMS = calc_RMS(sourceRotated, targetPsi)  % this is our loss function to minimize
     
    while (ii < maxIterations & (oldRMS-RMS > diffRMS))
        oldRMS = RMS;
        [~, targetPsi, ~, ~ ] = det_matching(sourceRotated, target);
        [R, t] = detRotation(sourceRotated, targetPsi, selectionType, sampleInd, nr_samples);
        sourceRotated = (R * sourceRotated' + t)';
        RMS = calc_RMS(sourceRotated, targetPsi)
        ii = ii + 1;
    end
    if ii == maxIterations
        message = 'maxIterations reached';
    elseif oldRMS-RMS < diffRMS
        message = ['convergence in: ', num2str(ii),  ' steps'];
    end 
end

function [R, t] = detRotation(source, psiTarget, selectionType, sampleInd, nr_samples)
    % determine R and t with SVD
    % source is the rotated source (N x 3)
    % psiTarget is the psiTarget (so closed points attached)
    %
    % dependent on selectionType the number of points are determined used
    % for calcuating the best R and t.
    % sampleInd only used if selectionType == 2
    % nr_samples only used if selectionType == 3
    switch selectionType
        case 1    % all points
            N = size(source, 1);
            P = source;
            Q = psiTarget;
        case 2    % given sample of points (sampleInd)
            N = length(sampleInd);
            P = source(sampleInd, :);
            Q = psiTarget(sampleInd, :);
        case 3    % every iteration new sample of size nr_samples
            Nmax = size(source, 1);
            N = nr_samples;          % parameter: number of samples
            sampleInd = randi(Nmax, N, 1);
            P = source(sampleInd, :);
            Q = psiTarget(sampleInd, :);
        otherwise
            disp('det_matching: this selectionType not supported!')
    end
    % step1 - determine weighted cetroids of p and q
    % note: weighting w = 1 for all points
    pc = sum(P, 1)/N;
    qc = sum(Q, 1)/N;
    % step2 - centered vectors -> matrices D x N
    X = (P - pc)';
    Xsum = sum(X, 2);   % check: should be 'zeroes'
    Y = (Q - qc)';
    Ysum = sum(Y, 2);   % check: should be 'zeroes'
    % step3 - determine covariance matrix S = XWY'
    W = eye(N);    % all weights are = 1
    S = X*W*Y';
    % step 4 - singular value decomposition
    [U Sigma V] = svd(S);
    s = svd(S);
    Nr = size(Sigma, 1);
    rot = eye(Nr);
    rot(Nr, Nr) = det(V*U');
    R = V*rot*U';
    % Step 5 - optimal translation
    t = qc' - R*pc';
end

function [psi, psiTarget, sampledSource, psiDistances] = det_matching(source, target)
    % source should be N x 3 matrix
    % target should be M x 3 matrix
    % psi is indices on target to map on source
    sampledSource = [];
    psi = [];
    psiTarget = [];
    psiDistances = [];
    N = size(source, 1);
    % N = 5   % for testing
    for ii = 1:N
        point = source(ii, :);
        [closestIndex, closestPoint, minDistance] = findClosestPoint(point, target);
        sampledSource = cat(1, sampledSource, point);
        psiTarget = cat(1, psiTarget, closestPoint);
        psi = cat(1, psi, closestIndex);
        psiDistances = cat(1, psiDistances, minDistance);
    end
end

function [closestIndex, closestPoint, minDistance] = findClosestPoint(point, target)
    % point is 1 x 3
    % target is N x 3
    % closest_point is 1 x 3
    % used brute force to determine the closest_point in target which
    % is closest to point. 
    distances = pdist2(target, point);
    minDistance = min(distances);
    closestIndex = find(distances == minDistance);
    closestPoint = target(closestIndex, :);
end

function [RMS] = calc_RMS (source, target)
    % RMS distance calculated between source and psi(target)
    % target is original set of points, psi give the corresponding 
    % indices of target-points which are closest to source
    N = size(source, 1);
    norms = sqrt(sum((source - target).^2,2));
    RMS = sqrt(sum(norms, 1)/N);
end
