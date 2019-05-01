% this has to run if to use vl toolbox
% first download toolbox
% run /home/peterheemskerk/vlfeat-0.9.21/toolbox/vl_setup;
% SK: Solved by adding VLFEATROOT to the gitfolder
% run vl_setup;

% 3. Fundamental Matrix, steps 1, 2 and 3
% find interest points and supposed matches between images
% for testing use boats TODO - use houses in stead
% use keypoint_matching.m as created in CV1Ass4.

% TODO eliminate back ground interest points
% QUESTION: column of V or column of V'
% QUESTION: not always column 9 in V ?
% QUESTION: inliers for all points or only for 8 test points

% create point of two images Ia, Ib,    
datapath = './Data/House/'
% name_image1 = 'boat1.pgm'
% name_image2 = 'boat2.pgm'
name_image1 = 'frame00000001.png'
name_image2 = 'frame00000041.png'
Ia = im2single(imread([datapath, name_image1]));
Ib = im2single(imread([datapath, name_image2]));
% graying not needed
 
%{
% just for fun, we show some of the selected matching keypoints. 
% for specific matches, presel has to be defined, otherwise presel = []
show=true
presel = [];   % note presel only has meaning if show = true
% presel = [3, 5, 6, 8, 15];     % these are some matches
thresh = 4;
[matches, fa, fb] = keypoint_matching(Ia, Ib, thresh, show, presel);
%}

% for serious business, we use key_point matching with no show.
% key_point_matching delivers matching interest points
disp('serious')
thresh = 4;
[matches, fa, fb] = keypoint_matching(Ia, Ib, thresh);
nr_matches = size(matches,2)
m1 = matches(1, :);
m2 = matches(2, :);
pointsO = fa(1:2, m1);
pointsM = fb(1:2, m2);
    
% make points homogenous
pointsO = cat(1, pointsO, ones(1, size(pointsO, 2)));
pointsM = cat(1, pointsM, ones(1, size(pointsM, 2)));

% testing (1) F with eight - point algorithm
F1 = eightPoint(pointsO, pointsM)           % normalisation = false default

% testing (2) F normalized eight point algorithm
F2 = eightPoint(pointsO, pointsM, true)       % normalisation = true

%{
% testing (3) RANSAC for determining Fundamental Matrix using eightPoint
% needs to be tuned, not very stable sofar in the returned Fundametal
% Matrix
n = 500;
p = 8;
thresh = 0.1;
F3 = fundRANSAC(pointsO, pointsM, n, p, thresh)        % default n, p, threshold
%}

%{
% epipolar - NOT YET READY
% take point - any point OR matching point ? -
% make it homogeneous
nrPoint = 25
pointEpiO = cat(1, pointsO(:, nrPoint), 1);
pointEpiM = cat(1, pointsM(:, nrPoint), 1);

epiPolar(Ia, Ib, pointEpiO, pointEpiM, F1)
%}

% =========================================
% calculate epipolar and make visible
% =========================================

function epiPolar(ImageO, ImageM, pointO, pointM, F)
% epipolar with F1 : line F*point
% assume point is homogenious: 3 X 1
[ymax, xmax] = size(ImageO);
epiLine = F * pointO
point1 = [0, ymax - epiLine(3)]
nx = xmax/epiLine(1);
ny = (ymax - epiLine(3))/epiLine(2);
% below only works is epiLine(3) > 0 and epiLine(1) and (2) < 0
if epiLine(3) > 0
    if (epiLine(1) < 0 & epiLine(2) < 0)
        if nx > ny
            point2 = [xmax, ymax - (epiLine(3) + nx*epiLine(2))]
        else
            point2 = [nx*epiLine(1), 0]
        end
    end
end
% plotting.
vl_tightsubplot(1,2,1);
% figure
imshow(ImageO)
hold on;
p = pointO(1:2);
plot(p(1), p(2), 'go')
plot(point1(1), point1(2), 'wo')
plot(point2(1), point2(2), 'wo')
line([point1(1), point2(1)], [point1(2), point2(2)] )
hold off;
vl_tightsubplot(1,2,2);
% figure
imshow(ImageM)
hold on;
m = pointM(1:2);
plot(m(1), m(2), 'yo')
plot(point1(1), point1(2), 'wo')
plot(point2(1), point2(2), 'wo')
line([point1(1), point2(1)], [point1(2), point2(2)] )
hold off;
end

