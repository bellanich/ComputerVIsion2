function [M, S] = motionStructure(D)

% step 1 - sample dense block - assumption that all is dense 
D;
% step 2 - point normalisation
mD = mean(D, 2);
sizemD=size(mD);
Dnorm = D-mD;
size(Dnorm);
% step 3 - SVD - and reduction to rank 3
[U, W, V] = svd(Dnorm);
U3 = U(:, 1:3);
W3 = W(1:3, 1:3);
V3 = V(:, 1:3) ;        
D3 = U3 * W3 * V3';
sizeD3 = size(D3);
% construct M for Motion and S for Structure
M = U3 * W3^(1/2);
S = W3^(1/2) * V3';
end




