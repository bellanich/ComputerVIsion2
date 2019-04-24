% % just to try out some code and get used to the data
% 
% % this is path where readPcd code is
% % readPcd code is amended: '\n' --> \r' Windows --> Linux
% addpath('./SupplementalCode'); 
% % please note...    ./ is current node
% 
% 
% datapath = './Data/';
% 
% s = load([datapath, 'source.mat']);
% 
% size_s = size(s.source, 2);
% 
% t = load([datapath, 'target.mat']);
% size_t = size(t.target);
% 
% a1 = readPcd([datapath, '0000000013_normal.pcd']);
% a2 = readPcd([datapath, '0000000000.pcd']);
% 
% figure
% X = s.source(1, :);
% Y = s.source(2, :);
% Z = s.source(3, :);
% C = zeros(size_s, 1);
% C = 1;
% scatter3(X, Y, Z, 0.7, 'r' )
% 
% figure
% size(a2)
% X1 = a2(:, 1);
% Y1 = a2(:, 2);
% Z1 = a2(:, 3);
% C1 = a2(:, 4);
% fscatter3(X1, Y1, Z1, C1)
% 
% figure
% a3 = a2(a2(:, 3)<1, :);
% size(a3)
% X2 = a3(:, 1);
% Y2 = a3(:, 2);
% Z2 = a3(:, 3);
% C2 = a3(:, 4);
% fscatter3(X2, Y2, Z2, C2)
% 
% R = load('R.mat')
% t = load('t.mat')
% 
% target =" readPcd([datapath, '0000000000.pcd']);
% source = readPcd([datapath, '0000000001.pcd']);
% size(source)
% size(R)
% sourceRot = [R*source'+t]'

openfig("Plots/Merged_mRMS");

