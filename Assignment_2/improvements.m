%{
1. PVM density improvements:
    This only works if you can somehow interpolate new points. Just
    recapturingthe the feature chains doesn't improve the selection of
    dense blocks. 

2. Affine ambiguity elimination
    Eliminating Affine ambiguity makes the point cloud reconstruction more
    reliable.

3. Improving the key-point matching
    By improving the key-point matching the features will be picked more
    reliably. This will translate into a denser point view matrix and a
    better 3D point reconstruction.

%}

test_M = 60 .* rand(8, 3) - 30;

row = M(1,:);