% just to try out some code and get used to the data

% addpath('./SupplementalCode')

datapath = '../../data/'

s = load('source.mat')
s
size(s.source)

t = load('target.mat')
size(t.target)

p = readPcd([datapath, '0000000013_normal.pcd'])
