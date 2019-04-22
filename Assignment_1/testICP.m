% Testing ICP on accuracy, stability hyperparameters etc.

clear all
clc
close all
% this is path where readPcd code is
% readPcd code is amended: '\n' --> \r' Windows --> Linux
addpath('./SupplementalCode') 
% please note...    ./ is current node
datapath = './Data/';


% open source files
% ICP method is tested on source.mat and target.mat

Asource = load([datapath, 'source.mat']);
Atarget = load([datapath, 'target.mat']);

% set ICP parameters

selectionType = 2;     % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                    
                        
nr_samples = 20;        % only used for selectionType = 2, 3 and 4
maxIterations = 300;    % max if no convergence
diffRMS = 0.0002;       % convergence if small improvement in RMS

listAvgRMS = [];
% list_nr_samples = [20 50 100 200 300 500 700 1000 1500]
list_nr_samples = [100 200]
for nr_samples = list_nr_samples

        nrTests = 20;
        reportSteps = [];
        reportRMS = [];
        for ii = 1:nrTests
    
            [RMS, message, R, t, listRMS, nrIterations] = ...
                ICP(Asource.source', Atarget.target', selectionType, nr_samples, maxIterations, diffRMS);
 
            % message
            reportSteps = [reportSteps, nrIterations];
            reportRMS = [reportRMS, RMS];

        end    
    
        nr_samples
        avgRMS = mean(reportRMS)
        stdRMS = std(reportRMS)
        avgSteps = mean(reportSteps)
        listAvgRMS = [listAvgRMS, avgRMS];

end

figure
plot(list_nr_samples, listAvgRMS)
title('Elbow')
xlabel('sample size')
ylabel('RMS')
axis([0 1500 0.15 0.23])

figure
plot(listRMS, '-o')
title('RMS per iteration')
xlabel('step')
ylabel('RMS')
