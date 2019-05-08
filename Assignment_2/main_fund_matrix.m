%% Testing eightPoint.m
clear
clc
close all

% Images to test
image1 = 'Data/House/frame00000001.png';
image2 = 'Data/House/frame00000041.png';

% =========== Plotting results ==================
%{     
                   Switch for F_type
        possible arguements are 'F1', 'F2', and 'F3'

        'F1' = eight-point algorithm fundamental matrix
        'F2' = normalized eight-point algorithm fundamental matrix
        'F3' = normalized eight-point algorithm with RANSAC fundamental
               matrix
%}

eightPoint_test(image1, image2, 'F3');