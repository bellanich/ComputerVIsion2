% just to try out some code and get used to the data

% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node

datapath = '../../data/'

s = load('source.mat')

size_s = size(s.source, 2)

t = load('target.mat')
size_t = size(t.target)

a1 = readPcd([datapath, '0000000013_normal.pcd']);
a2 = readPcd([datapath, '0000000000.pcd']);

X = s.source(1, :);
Y = s.source(2, :);
Z = s.source(3, :);
C = zeros(size_s, 1);
C = 1;

scatter3(X, Y, Z, 0.7, 'r' )

% fscatter3(X, Y, Z, C)