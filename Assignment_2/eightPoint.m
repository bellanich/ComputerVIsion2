function F = eightPoint(pointsO, pointsM, normalise)
% pointsO and pointsM should be homogeneous: 3 X N matrices
 
if nargin == 2
    normalise = false
end

if normalise
    disp ('normalise')
else
    disp ('no normalisation')
    F = fundamentalMatrix(pointsO, pointsM)
end
end


% ===============================================================
%   helper functions
% ===============================================================

function F = fundamentalMatrix(pointsO, pointsM)
    disp('fundamental matrix calc')
    noPoints = size(pointsO, 2)
    % step 1 - construct matrix A
    A = zeros(noPoints, 9);
    for ii = 1:noPoints
        xO = pointsO(1, ii);
        yO = pointsO(2, ii);
        xM = pointsM(1, ii);
        yM = pointsM(2, ii);
        A(ii, 1) = xO*xM;
        A(ii, 2) = xO*yM;
        A(ii, 3) = xO;
        A(ii, 4) = yO*xM;
        A(ii, 5) = yO*yM;
        A(ii, 6) = yO;
        A(ii, 7) = xM;
        A(ii, 8) = yM;
        A(ii, 9) = 1;        
    end
    % step 2 - SVD
    [U, D, V] = svd(A);
    % step 3 - take column with smallest singular value
    % this is normaly column 9, but if number of points < 9, take that. 
    minCol = min(9, noPoints)
    D
    sizeV = size(V)
    Fentries = V(:, minCol)       % TODO - make sure V and not V'
    F = reshape(Fentries, [3,3])  % TODO - make sure this is always correct
end