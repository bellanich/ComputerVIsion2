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



%% 

load stereoPointPairs
[fLMedS,inliers] = estimateFundamentalMatrix(matchedPoints1,...
    matchedPoints2,'NumTrials',4000);

I1 = imread('viprectification_deskLeft.png');
figure; 
subplot(121);
imshow(I1); 
title('Inliers and Epipolar Lines in First Image'); hold on;
plot(matchedPoints1(inliers,1),matchedPoints1(inliers,2),'go')

epiLines = epipolarLine(fLMedS',matchedPoints2(inliers,:));
points = lineToBorderPoints(epiLines,size(I1));
line(points(:,[1,3])',points(:,[2,4])');

I2 = imread('viprectification_deskRight.png');
subplot(122); 
imshow(I2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(matchedPoints2(inliers,1),matchedPoints2(inliers,2),'go')
