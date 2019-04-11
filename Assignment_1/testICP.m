% Testing ICP on rubustness, hyperparameters etc.

clear all
clc
close all
% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node
datapath = './Data/';


% open source files
% ICP method is tested on source.mat and target.mat

Asource = load([datapath, 'source.mat']);
Atarget = load([datapath, 'target.mat']);


% set ICP parameters

selectionType = 2;      % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                        % 4 is NOT implemented yet
                        
nr_samples = 1000;        % only used for selectionType = 2 or 3
maxIterations = 100;
diffRMS = 0.0005;       % convergence if small improvement in RMS

[RMS, message, Rot, trans] = ...
    ICP(Asource.source', Atarget.target', selectionType, nr_samples, maxIterations, diffRMS)


