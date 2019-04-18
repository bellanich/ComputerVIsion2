
function [RMS, message, R, t] = ...
    ICP (source, target, selectionType,  nr_samples, maxIterations, diffRMS)
   
    % Initilize R, tr, RMS
    dim = 3;
    R = eye(dim);
    t = zeros(dim, 1);
    RMS = 1000000000000;   
    oldRMS = 2*RMS;
    ii = 1;
    
    % initialze sample
    if selectionType == 1    % all points
        sourceSample = source;
    elseif selectionType == 2   % random points, fixed for all iterations
        sampleInd = selectRandom(source, nr_samples);
        sourceSample = source(sampleInd, :);
    elseif selectionType == 4  % points from densest clusters
        %  ------ Parameters to change ----------
        K = 25; %= nr_samples;
        nDensestClusters = 3; % sample from n densest cluster
    
        %  ----- Redefining source as only the points from the densest clusters ------
        [idx, clusterMeans]= kmeans(source, K);
        % Identify the densest clusters
        pointsPerCluster = histcounts(idx);
        [B, clustersToSample] = maxk(pointsPerCluster, nDensestClusters);
        % Pull datapoints from all denest clusters and rename as Source2
        densePoints_idx = find(ismember(idx, clustersToSample));
        source2 = source(densePoints_idx,:);
      
        %  ----- Randomly selecting samples from redfined source ------
        sourceSample = [];
        for sample = 1:nr_samples
            % Randomly sample a single datapoint from source2
            datapoint = source2(randi(size(source2,1)),:);
            sourceSample = [sourceSample; datapoint]; 
        end
    end
        
    while (ii < maxIterations & (oldRMS-RMS > diffRMS | RMS > oldRMS))
        ii = ii + 1;
        if selectionType == 3   % new sample points for each iteration
            sampleInd = selectRandom(source, nr_samples);
            sourceSample = source(sampleInd, :);
        end
        
        % Rotate
        sourceRotatedSample = (R * sourceSample' + t)';
        % Match
        [~, targetPsi, ~ , ~ ] = det_matching(sourceRotatedSample, target);        
        % New RMS
        oldRMS = RMS;
        RMS = calc_RMS(sourceRotatedSample, targetPsi);
        % New R and t
        [R, t] = detRotation(sourceSample, targetPsi);
    end
    if ii == maxIterations
        message = 'maxIterations reached';
    elseif oldRMS-RMS < diffRMS
        message = ['convergence in: ', num2str(ii),  ' steps'];
    end 
end

function [sampleInd] = selectRandom(source, nr_samples)
    Nmax = size(source ,1);
    N = nr_samples;
    sampleInd = randi(Nmax, N, 1);
end

function [R, t] = detRotation(source, psiTarget)
    % determine R and t with SVD
    % source is the sampled source (N x 3)
    % psiTarget is the psiTarget (so closed points attached)
    %    
    N = size(source, 1);
    P = source;
    Q = psiTarget;
    % step1 - determine weighted centroids of p and q
    % note: weighting w = 1 for all points
    pc = sum(P, 1)/N;
    qc = sum(Q, 1)/N;
    % step2 - centered vectors -> matrices D x N
    X = (P - pc)';
    Xsum = sum(X, 2);   % check: should be 'zeroes'
    Y = (Q - qc)';
    Ysum = sum(Y, 2);   % check: should be 'zeroes'
    % step3 - determine covariance matrix S = XWY'
    W = eye(N);    % all weights are = 1
    S = X*W*Y';
    % step 4 - singular value decomposition
    [U, Sigma, V] = svd(S);
    s = svd(S);
    Nr = size(Sigma, 1);
    rot = eye(Nr);
    rot(Nr, Nr) = det(V*U');
    R = V*rot*U';
    % Step 5 - optimal translation
    t = qc' - R * pc' ;
end

function [psi, psiTarget, sampledSource, psiDistances] = det_matching(source, target)
    % source should be N x 3 matrix
    % target should be M x 3 matrix
    % psi is indices on target to map on source
    sampledSource = [];
    psi = [];
    psiTarget = [];
    psiDistances = [];
    
    N = size(source, 1);
    % N = 5   % for testing
    for ii = 1:N
        point = source(ii, :);
        [closestIndex, closestPoint, minDistance] = findClosestPoint(point, target);
        sampledSource = cat(1, sampledSource, point);
        psiTarget = cat(1, psiTarget, closestPoint);
        psi = cat(1, psi, closestIndex);
        psiDistances = cat(1, psiDistances, minDistance);
    end            
end

function [closestIndex, closestPoint, minDistance] = findClosestPoint(point, target)
    % point is 1 x 3
    % target is N x 3
    % closest_point is 1 x 3
    % used brute force to determine the closest_point in target which
    % is closest to point. 
    distances = pdist2(target, point);
    minDistance = min(distances);
    closestIndex = min(find(distances == minDistance));    % min if more points are closest
    closestPoint = target(closestIndex, :);
end

function [RMS] = calc_RMS (source, target)
    % RMS distance calculated between source and psi(target)
    % target is original set of points, psi give the corresponding 
    % indices of target-points which are closest to source
    N = size(source, 1);
    norms = sqrt(sum((source - target).^2,2));
    RMS = sqrt(sum(norms, 1)/N);
end
