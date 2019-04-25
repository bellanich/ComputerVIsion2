function tform = RANSAC(coords1, coords2, n, p, threshold)
%RANSAC Performs RANSAC algorithm to find coordinate transformation
% IN
%  coords1 : coordinates to be transformed from
%  coords2 : coordinates to be transformed to
%  n : number of tries until best transformation is returned
%  p : size of random subset for which transformation is solved
%  threshold : threshold distance for determining translation correctness
% OUT
%  tform : transformation matrix

if nargin < 3
    n = 50;
end
if nargin < 4
    p = ceil(sqrt(length(coords1) - 1)) + 1;
end
if nargin < 5
    threshold = 10;
end

[ndims, ncoords, ~] = size(coords1);
p = min(p, ncoords);

% create A matrix and b vector
A = createA(coords1);
b = coords2;
b = reshape(b, numel(b), 1);

tform = eye(ndims + 1);
tformCur = eye(ndims + 1);

% make coordinates homogeneous
coords1 = vertcat(coords1, ones(1, ncoords));
coords2 = vertcat(coords2, ones(1, ncoords));

sel = zeros(2 * p, 1);
bestScore = -1;
for ii = 1:n
    
    % create selection
    perm = randperm(ncoords);
    sel(1:p) = perm(1:p) * 2;
    sel(p+1:end) = sel(1:p) - 1;
    
    % solve for random subsets of A and b
    v = linsolve(A(sel, :), b(sel, :));
    
    % create new transformation matrix
    tformCur(1:ndims, 1:ndims) = ...
        transpose(reshape(v(1:ndims ^ 2), ndims, ndims));
    tformCur(1:ndims, end) = v(end - ndims + 1:end);
    
    % score the current M matrix and t vector
    coordsDiff = tformCur * coords1 - coords2;
    coordsDist = sqrt(sum(coordsDiff(1:end - 1, :) .^ 2));
    score = sum(coordsDist < threshold);
    
    % update transformation matrix if a better one is found
    if score > bestScore || bestScore < 0
        tform = tformCur;
        bestScore = score;
    end
    
end

end


function coordstack = createA(coords)
%CREATECOORDSTACK Creates A matrix for affine transform
% IN
%  coords : input coordinates

[ndims, ncoords, ~] = size(coords);

coordstack = zeros(ndims * ncoords, ndims * (ndims + 1));

atLast = ndims .^ 2 + 1;

% stack pairs for each coordinate
for ii = 0:ncoords-1
    for jj = 1:ndims
        coordstack(ndims * ii + jj, (jj - 1) * ndims + 1:jj * ndims) = ...
            transpose(coords(1:ndims, ii + 1));
    end
    coordstack(ndims * ii + 1:ndims * (ii + 1), atLast:end) = eye(ndims);
end

end