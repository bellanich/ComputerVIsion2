%% Performing RANSAC on pair of images
clear
clc
close all

im1 = imread('Data/House/frame00000001.png');
im2 = imread('Data/House/frame00000002.png');

% Keypoint Matching
visualize = true;
[f1, f2] = keypoint_matching(im1, im2, visualize);

% RANSAC
tform = RANSAC(f1(1:2, :), f2(1:2, :));

%% To visualize transformed images
figure();

subplot(211);
imshow(im1);
title('im1');

subplot(212);
imshow(im2);
title('im2');

subplot(223);
imshow(nearestNeighbourImwarp(im1, tform));
title('im1 -> im2');

subplot(224);
imshow(nearestNeighbourImwarp(im2, tform \ eye(size(tform))));
title('im2 -> im1');

