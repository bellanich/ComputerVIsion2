function [fmatch1, fmatch2] = keypoint_matching(...
    im1, im2, ...
    threshPeak, threshEdge, threshMatch)
%KEYPOINT_MATCHING Finds matching features between input images using SIFT
% IN
%  im1 : first input image
%  im2 : second input image
%  threshPeak : SIFT peak threshold
%  threshEdge : SIFT edge threshold
%  threshMatch : descriptor matching threshold
% OUT
%  fmatch1 : features from first image
%  fmatch2 : features from second image

% Assigning variable values to unspecified vars
if nargin < 3
    threshPeak = 0;
end
if nargin < 4
    threshEdge = 10;
end
if nargin < 5
    threshMatch = 1.5;
end


% Preprocessing images
im1 = im2gray255(im1);
im2 = im2gray255(im2);

% Obtaining features and descriptors
[f1, d1] = vl_sift(im1, ...
    'PeakThresh', threshPeak, 'edgethresh', threshEdge);
[f2, d2] = vl_sift(im2, ...
    'PeakThresh', threshPeak, 'edgethresh', threshEdge);

% Obtaining matching feature indices
[matches, ~] = vl_ubcmatch(d1, d2, threshMatch);

% Obtaining matching features
fmatch1 = f1(:, matches(1, :));
fmatch2 = f2(:, matches(2, :));

end


function imOut = im2gray255(im)
%IM2GRAY255 Converts image to grayscale in range 0-255
% IN
%  im : input image
% OUT
% imOut : output image

[~, ~, c, ~] = size(im);

% convert images to grayscale
if c > 1
    im = rgb2gray(im);
end

% convert images to range 0-255
if max(max(im)) <= 1
    im = 255 * im;
end

imOut = single(uint8(im));

end