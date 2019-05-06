function imOut = nearestNeighbourImwarp(im, tform)
%NEARESTNEIGHBOURIMWARP Creates a warped image, using nearest neighbour
%  interpolation to build new image
% IN
%  im : original image
%  tform : transformation matrix
% OUT
%  imOut : output image

[h, w, c, ~] = size(im);

% determine offsets
corners = tform * [0, w, 0, w; 0, 0, h, h; 1, 1, 1, 1];
offsets = round(min(corners, [], 2));
offsets(end) = 0;

% determine size of output image
outsize = round(max(corners - offsets, [], 2));
imOut = zeros(outsize(2), outsize(1), c, class(im));

% get output image coordinate grid
[Y, X] = meshgrid(1:outsize(2), 1:outsize(1));
ncoordsOut = numel(X);
coordsOut = [reshape(X, 1, ncoordsOut); reshape(Y, 1, ncoordsOut); ...
    ones(1, ncoordsOut)];

% calculate input image grid equivalent
coordsIn = round((tform \ eye(size(tform))) * (coordsOut + offsets));

% create selection of valid coordinates
sel = (min(coordsIn(1:2, :) > 0) + (coordsIn(1, :) <= w) ...
    + (coordsIn(2, :) <= h) + 0) == 3;
selXOut = coordsOut(1, sel);
selYOut = coordsOut(2, sel);
selXIn = coordsIn(1, sel);
selYIn = coordsIn(2, sel);

% set output image values
for ii = 1:length(selYOut)
    imOut(selYOut(ii), selXOut(ii), :) = im(selYIn(ii), selXIn(ii), :);
end

end