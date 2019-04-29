function F = eightPoint(pointsO, pointsM, normalise)
% pointsO and pointsM should be homogeneous: 3 X N matrices
 
if nargin == 2
    normalise = false;
end

if normalise
    disp ('normalise')
    % normalise
    phatO = normalize(pointsO);
    phatM = normalize(pointsM);
    F = fundamentalMatrix(phatO, phatM);
else
    disp ('no normalisation')
    F = fundamentalMatrix(pointsO, pointsM);
end
end


% ===============================================================
%   helper functions
% ===============================================================

function phat = normalize(points)
    noPoints = size(points, 2);
    % normalise
    m = mean(points, 2)
    d = 0;
    for ii = 1:noPoints
        d = d + norm(points - m);
    end
    d = d / noPoints;
    T = [sqrt(2)/d 0 -m(1)*sqrt(2)/d; ...
        0 sqrt(2)/d -m(2)*sqrt(2)/d; ...
        0 0 1];
    phat = T * points;
end

function F = fundamentalMatrix(pointsO, pointsM)
    % disp('fundamental matrix calc');
    noPoints = size(pointsO, 2);
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
    D;
    sizeV = size(V);
    Fentries = V(:, minCol);       % TODO - make sure V and not V'
    Fnonsing = reshape(Fentries, [3,3]);  % TODO - make sure this is always correct
    % step 4 - ensure singularity of F
    [Uf, Df, Vf] = svd(Fnonsing);
    Df(3, 3) = 0;
    F = Uf * Df * Vf';
    % test that F is different from Fnonsing
    Fnonsing - F;
end