
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

%% Matlab example
% Plot epipolar lines generated from matrix
load stereoPointPairs

% load stereoPointPairs
[fLMedS,inliers] = estimateFundamentalMatrix(matchedPoints1,...
    matchedPoints2,'NumTrials',4000);

I1 = imread('viprectification_deskLeft.png');
figure; 
subplot(121);
imshow(I1); 
title('Inliers and Epipolar Lines in First Image'); hold on;
plot(matchedPoints1(inliers,1),matchedPoints1(inliers,2),'go')

epiLines = epipolarLine(fLMedS',matchedPoints2(inliers,:));
points = lineToBorderPoints(epiLines, size(I1));
line(points(:,[1,3])',points(:,[2,4])');

I2 = imread('viprectification_deskRight.png');
subplot(122); 
imshow(I2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(matchedPoints2(inliers,1),matchedPoints2(inliers,2),'go')
