
function [M, S] = motionStructure(Dsparse, base)
% function delivers M and S matrices 

if nargin == 1
    base = 0;
end

% step 0 - sample dense block - assumption that all is dense 
% look for points which are in 90% (parameter) of projections
threshold = 0.9;
D = [];
[noProjections, noPoints] = size(Dsparse);
for ii = 1:noPoints
    points = Dsparse(:, ii);
    if sum(points(points(1) > 0)) / noProjections > threshold
        D = cat(2, D, points);
    end
end

% step: process whole D (if base == 0) or walk through D with steps = base
[noProjections, noPoints] = size(D);
if base == 0
    [M, S] = calcMS(D);
else
    step = 2*base;
    noIterations = floor(noPoints/step);
    for ii = 1:noIterations
        Diter = D((ii-1)*step+1:ii*step, :);
        [Miter, Siter] = calcMS(Diter);
        if ii == 1
            Sstart = Siter;
            S = Sstart;
            sizeStart = size(S)
        else
            [~,Z] = procrustes(Sstart,Siter);
            S = cat(2, S, Z);
        end
    end
    sizeS = size(S)
    M = 0
end
end

% ===============================================================
% helper function
% ===============================================================

function [M, S] = calcMS(D)
% D is dense D 
    % step 1 - point normalisation
    mD = mean(D, 2);
    sizemD=size(mD);
    Dnorm = D-mD;
    % step 2 - SVD - and reduction to rank 3
    [U, W, V] = svd(Dnorm);
    U3 = U(:, 1:3);
    W3 = W(1:3, 1:3);
    V3 = V(:, 1:3) ;        
    D3 = U3 * W3 * V3';
    % step 3 - construct M for Motion and S for Structure
    M = U3 * W3^(1/2);
    S = W3^(1/2) * V3';
end

