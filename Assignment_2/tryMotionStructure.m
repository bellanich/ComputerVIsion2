% Open PointViewMatrix

clear all
clc
close all

%{
addpath('./SupplementalCode') 
% please note...    ./ is current node
datapath = './Data/';
%}

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

sizeP = size(P)

% M, S are based on full PointViewMatrix
[M, S] = motionStructure(P);
sizeM = size(M)
sizeS = size(S)
% plotting these, we see the roof of a house - or is it a buffalo :)
plot_3D(S);

% M, S are based on three images only
base= 3
[~, S3] = motionStructure(P, base);
sizeS3 = size(S3)
plot_3D(S3)

% M, S are based on four images only
base = 4
[~, S4] = motionStructure(P, base);
sizeS4 = size(S4)
plot_3D(S4)

% M, S are based on four images only
base = 10
[~, S10] = motionStructure(P, base);
sizeS4 = size(S10)
plot_3D(S10)

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