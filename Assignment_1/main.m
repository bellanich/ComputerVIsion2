%% Testing readPcd.m and reading source and target data

clear all
clc
close all

% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node
datapath = './Data/';

source = load([datapath, 'source.mat']);
size(source.source);
target = load([datapath, 'target.mat']);
size(target.target);

%{
BaseCloud = readPcd([datapath, '0000000013_normal.pcd']);
TargetCloud = readPcd([datapath, '0000000000.pcd']);
%}

%% ICP method is tested on source.mat and target.mat

selectionType = 3;      % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                        % 4 is NOT implemented yet
                        
nr_samples = 25;        % only used for selectionType = 2 or 3
maxIterations = 100;
diffRMS = 0.0005;       % convergence if small improvement in RMS

[RMS, message, Rot, trans] = ICP(source.source', target.target', selectionType, nr_samples, maxIterations, diffRMS);
