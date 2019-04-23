% Testing ICP on accuracy, stability hyperparameters etc.
%% Initalize file to store data
fileID = fopen('ICP_testing.txt','w');

%%
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
selectionType_list = [2, 3, 4];
nr_sample_list = [20, 100, 300, 1000];

% selectionType = 1;     % 1 = use all the points (a)
                        % 2 = sample subset of points (b)
                        % 3 = sample subset of points every iteration (c)
                        % 4 = sample from points of interest (d)
                    
                        
% nr_samples = 4;        % only used for selectionType = 2, 3 and 4
maxIterations = 300;    % max if no convergence
diffRMS = 0.0005;       % convergence if small improvement in RMS


for selectionType_index = 1:length(selectionType_list) 
    for nr_samples_index = 1:length(nr_sample_list)
        
        selectionType = selectionType_list(selectionType_index);
        nr_samples = nr_sample_list(nr_samples_index);
    
        
        listAvgRMS = [];
        % list_nr_samples = [20 50 100 200 300 500 700 1000 1500]
        %list_nr_samples = [100 200];
        nrTests = 20;
        reportSteps = [];
        reportRMS = [];
        
        ICP_count = 0;
        for ii = 1:nrTests
            
            [RMS, message, R, t, listRMS, nrIterations] = ...
                ICP(Asource.source', Atarget.target', selectionType, nr_samples, maxIterations, diffRMS);

            % message
            reportSteps = [reportSteps, nrIterations];
            reportRMS = [reportRMS, RMS];
            
            ICP_count = ICP_count + 1;

        end    
        
        fileID = fopen('ICP_testing.txt','a');
        nr_samples;
        avgRMS = mean(reportRMS);
        stdRMS = std(reportRMS);
        avgSteps = mean(reportSteps);
        listAvgRMS = [listAvgRMS, avgRMS];
        end_time = clock;
        run_time = datestr(etime(end_time,start_time),'HH:MM:SS:FFF');
        
        my_string = ['avgRMS = ', num2str(avgRMS), ', ', 'stdRMS = ', num2str(stdRMS), ', ', 'avgSteps = ', num2str(avgSteps), ', ', 'ICP_count = ', num2str(ICP_count), ', ', 'run_time = ', num2str(run_time)];
        fprintf(fileID,'%10s \n', my_string);
        fclose(fileID);



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
   end
end

