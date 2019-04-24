% Testing ICP with added Noise.

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

selectionType = 1;     % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                    
                        
nr_samples = 200;        % only used for selectionType = 2, 3 and 4
maxIterations = 300;    % max if no convergence
diffRMS = 0.0005;       % convergence if small improvement in RMS

listPercNoise = [0.0 0.01 0.02 0.05 0.1];
listAvgRMSNoise = [];
progressbar('percNoise', 'ICP', 'Tests')
iter = 0;
for percNoise = listPercNoise

    nrNoise = round(size(Asource.source', 1) * percNoise);
    NoisedSource = addNoise(Asource.source', nrNoise);

    listAvgRMS = [];


        nrTests = 10;
        reportSteps = [];
        reportRMS = [];
        for ii = 1:nrTests
    
            [RMS, message, R, t, listRMS, nrIterations] = ...
                ICP(NoisedSource, Atarget.target', selectionType, nr_samples, maxIterations, diffRMS);
       
            % [RMS, message, R, t, listRMS, nrIterations] = ...
            %     ICP(Asource.source', Atarget.target', selectionType, nr_samples, maxIterations, diffRMS);
 
            % message
            reportSteps = [reportSteps, nrIterations];
            reportRMS = [reportRMS, RMS];
            
            progressbar([], [], ii/nrTests)

        end    
    
        avgRMS = mean(reportRMS);
        stdRMS = std(reportRMS);
        avgSteps = mean(reportSteps);
        listAvgRMS = [listAvgRMS, avgRMS];

    listAvgRMSNoise = [listAvgRMSNoise, avgRMS];
    
    iter = iter + 1;
    progressbar(iter / 5)
   
end

figure
plot(listPercNoise, listAvgRMSNoise)
title('Noise dependency')
xlabel('noise percentage')
ylabel('RMS')
axis([0 0.1 0.15 0.35])

function NoisedSource = addNoise(source, nrNoise)
    % Bsource = Asource.source';     % use for adding noise
    % percNoise = 0.1
    % nrNoise = round(size(Bsource, 1) * percNoise)
    % size(Bsource)
    % min(Bsource);
    % max(Bsource);
    % max(Bsource) - min(Bsource);
    N = rand(nrNoise,3) .* (max(source)-min(source)) + min(source);
    NoisedSource = cat(1, source, N);
    % size(Nsource)
end