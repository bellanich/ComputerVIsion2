function bestF = fundRANSAC(coords1, coords2, n, p, threshold)
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
p = 8


% bestF = eye(ndims + 1);
% tformCur = eye(ndims + 1);

% make coordinates homogeneous - TODO only if necessary:
if size(coords1, 1) < 3
    coords1 = vertcat(coords1, ones(1, ncoords));
    coords2 = vertcat(coords2, ones(1, ncoords));
end

% sel = zeros(2 * p, 1);
bestScore = -1;
for ii = 1:n
    
    % create selection
    % perm = randperm(ncoords);
    % sel(1:p) = perm(1:p) * 2;
    % sel(p+1:end) = sel(1:p) - 1;
    sel = randperm(ncoords, p)
    
    % solve for selection
    Fcur = eightPoint(coords1(:, sel), coords2(:, sel), true)  % 8 points
    
    % score the current M matrix and t vector
    coordsDist = distanceSampson(Fcur, coords1, coords2);   % all points
    score = sum(coordsDist < threshold)
    
    % update transformation matrix if a better one is found
    if score > bestScore || bestScore < 0
        bestF = Fcur;
        bestScore = score;
    end
    %}
end

end

% ===================================================================
%   helper
% ===================================================================

function ds = distanceSampson (F, pointsO, pointsM)
    % points should be homogeneous: 3 x N  (3rd row ones)
    ds = [];
    noPoints = size(pointsO, 2);
    for ii = 1:noPoints
        pO = pointsO(:, ii);
        pM = pointsM(:, ii);
        da = (pM'*F*pO)^2;   % should be scalar
        FpO = F*pO;
        FpM = F*pM;
        db = FpO(1)^2 + FpO(2)^2 + FpM(1)^2 + FpM(2)^2; % should be scalar
        ds = [ds da/db];      % added to the list
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