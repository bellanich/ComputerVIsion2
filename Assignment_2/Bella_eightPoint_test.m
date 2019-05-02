clear
clc
close all

%  ============ Set up information for plotting  ==================

% Load images
image1 = imread('Data/House/frame00000001.png');
image2 = imread('Data/House/frame00000041.png');
Ia = im2single(image1);
Ib = im2single(image2);

% Parameters
thresh = 5;
[matches, fa, fb] = keypoint_matching(Ia, Ib, thresh);
nr_matches = size(matches,2);
m1 = matches(1, :);
m2 = matches(2, :);
pointsO = fa(1:2, m1);
pointsM = fb(1:2, m2);
pointsO = cat(1, pointsO, ones(1, size(pointsO, 2)));
pointsM = cat(1, pointsM, ones(1, size(pointsM, 2)));

% =========== Plotting results ==================
%{     
                   Switch for F_type

        possible arguements are 'F1', 'F2', and 'F3'
%}
epipolar_plot(image1, image2, pointsO, pointsM, 'F3') 

function [] = epipolar_plot(image1, image2, pointsO, pointsM, F_type)

% Determining F_type to use
if F_type == 'F1'
    F = eightPoint(pointsO, pointsM);
elseif F_type == 'F2'
    F = eightPoint(pointsO, pointsM, true);
elseif F_type == 'F3'
    n = 500;
    p = 8;
    thresh = 0.5;
    F = fundRANSAC(pointsO, pointsM, n, p, thresh); 
end

% plots feature-matching points for image 1
figure; 
subplot(121);
imshow(image1); hold on;
title('Inliers and Epipolar Lines in First Image');
plot(pointsO(1,:), pointsO(2,:), 'go')

% Find a way to plot image2 points onto image 1???
% % showing epipolar lines of image 1
% epiLines = epipolarLine(F1, pointsM');
% points = lineToBorderPoints(epiLines,size(image1));
% line(points(:,[1,3])',points(:,[2,4])');

% plots feature-matching points for image 2
subplot(122); 
imshow(image2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(pointsM(1,:), pointsM(2,:),'go')

% showing epipolar lines on image2
epiLines = epipolarLine(F,(pointsO(1:2, :))');
points = lineToBorderPoints(epiLines,size(image2));
line(points(:,[1,3])', points(:,[2,4])');
%truesize;
end