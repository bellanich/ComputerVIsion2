function [] = eightPoint_test(image1, image2, F_type)
%{
    :params:
    image1 = filepath for first image
    image 2 = filepath for second image

    F_type: method for fundamental matrix that you want to use,
    possible arguements are: 'F1', 'F2', and 'F3'
%}

% Loading images
image1 = imread(image1);
image2 = imread(image2);

Ia = im2single(image1);
Ib = im2single(image2);

% Performing keypoint matching
thresh = 5;
[matches, fa, fb] = keypoint_matching(Ia, Ib, thresh);

% Reformating results from keypoint matching to get pointsO and pointsM
nr_matches = size(matches,2);
m1 = matches(1, :);
m2 = matches(2, :);
pointsO = fa(1:2, m1);
pointsM = fb(1:2, m2);
pointsO = cat(1, pointsO, ones(1, size(pointsO, 2)));
pointsM = cat(1, pointsM, ones(1, size(pointsM, 2)));

% Calling helper function to plot results
epipolar_plot(image1, image2, pointsO, pointsM, F_type) 
end

% ================== Helper Function ===================
% Used to generate the epipolar lines and plot them on the image
function [] = epipolar_plot(image1, image2, pointsO, pointsM, F_type)

% Determining F_type to use
if F_type == 'F1'
    F = eight_Point(pointsO, pointsM);
elseif F_type == 'F2'
    F = eight_Point(pointsO, pointsM, true);
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
