%% RANSAC test on a pair of images
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


%% Testing eightPoint.m
clear
clc
close all

%datapath = './Data/House/';
image1 = imread('Data/House/frame00000001.png');
image2 = imread('Data/House/frame00000041.png');
Ia = im2single(image1);
Ib = im2single(image2);

% disp('serious')
thresh = 5;
[matches, fa, fb] = keypoint_matching(Ia, Ib, thresh);
nr_matches = size(matches,2);
m1 = matches(1, :);
m2 = matches(2, :);
pointsO = fa(1:2, m1);
pointsM = fb(1:2, m2);

F1 = eightPoint(pointsO, pointsM);

% plots points for image 1
figure; 
subplot(121);
imshow(image1); hold on;
title('Inliers and Epipolar Lines in First Image');
plot(pointsO(1,:), pointsO(2,:), 'go')

% showing epipolar lines of image 1
epiLines = epipolarLine(F1, pointsM');
points = lineToBorderPoints(epiLines,size(image1));
line(points(:,[1,3])',points(:,[2,4])');

% plots points for image 2
subplot(122); 
imshow(image2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(pointsM(1,:), pointsM(2,:),'go')

% showing epipolar lines on image2
epiLines = epipolarLine(F1,pointsO');
points = lineToBorderPoints(epiLines,size(image2));
line(points(:,[1,3])',points(:,[2,4])');
truesize;
