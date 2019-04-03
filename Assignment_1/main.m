%% Testing readPcd.m 
clear all
clc
close all

BaseCloud = readPcd('0000000000.pcd');
TargetCloud = readPcd('0000000001.pcd');
%% ICP

% Initilize 3 by 3 identity matrix

% Create function to search for all points in the base cloud to find the 
% closest point in the target cloud. 
% Will call this psi.
% Need to output a matrix of pairwise combinations (base point, nearest target point)

% THEN update approach, to get R and t using SVD.
% create if else statements to see when RMS changes