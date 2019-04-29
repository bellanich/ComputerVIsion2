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

[M, S] = motionStructure(P);

sizeM = size(M)
sizeS = size(S)