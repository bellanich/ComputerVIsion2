% Testing ICP on accuracy, stability hyperparameters etc.
%% Initialize file to store data
% Only run if you want to overwrite previously existing file
fileID = fopen('ICP_testing.txt','w');

%% Run ICP for particular selectionType, and particular nr_samples

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
selectionType_list = [1, 2, 3, 4];
nr_samples_list = [20, 100, 300, 1000];

% What selection type corresponds to what
% 1 = use all the points (a)
% 2 = sample subset of points (b)
% 3 = sample subset of points every iteration (c)
% 4 = sample from points of interest (d)
                    
                        
% nr_samples = 4;        % only used for selectionType = 2, 3 and 4
maxIterations = 300;    % max if no convergence
diffRMS = 0.0005;       % convergence if small improvement in RMS


for selectionType_index = 1:length(selectionType_list) 
    
    listAvgRMS = [];
    for nr_samples_index = 1:length(nr_samples_list)
        
        % unpacking list as variables that we'll loop over
        selectionType = selectionType_list(selectionType_index);
        nr_samples = nr_samples_list(nr_samples_index);
        
        % setting start time (to calculate time it takes to run code)
        start_time = datestr(clock,'YYYY/mm/dd HH:MM:SS:FFF');
        
        % list_nr_samples = [20 50 100 200 300 500 700 1000 1500]
        %list_nr_samples = [100 200];
        nrTests = 20;
        reportSteps = [];
        reportRMS = [];
        
        % Tracks the number of ICP iterations obtained
        

        for ii = 1:nrTests
            
            [RMS, message, R, t, listRMS, nrIterations] = ...
                ICP(Asource.source', Atarget.target', selectionType, nr_samples, maxIterations, diffRMS);
           
            % message
            reportSteps = [reportSteps, nrIterations];
            reportRMS = [reportRMS, RMS];
            

        end    
        
        
        nr_samples;
        
        % ---------- calculate save results ----------------
        % calculate results
        avgRMS = mean(reportRMS);
        stdRMS = std(reportRMS);
        avgSteps = mean(reportSteps);
        % end time
        end_time = datestr(clock,'YYYY/mm/dd HH:MM:SS:FFF');
        listAvgRMS = [listAvgRMS, avgRMS];
        % save in .txt file
        fileID = fopen('ICP_testing.txt','a');
        value_labels = ['For the run with selection type = ', num2str(selectionType), ' , nr_samples = ', num2str(nr_samples)];
        result_values = ['avgRMS = ', num2str(avgRMS), ', ', 'stdRMS = ', num2str(stdRMS), ', ', 'avgSteps = ', num2str(avgSteps), ', ', 'start_time = ', num2str(start_time), ', ', 'end_time = ', num2str(end_time)];
        results = [value_labels, newline, result_values, newline];
        fprintf(fileID,'%10s \n', results);
        fclose(fileID);

        % sample number doesn't affect selectionType one, so break this
        % loop
        if selectionType == 1
            break;
        end
        

    end
     % ------------ Plotting results for each selection Type --------------------
     
     % average RMS as a function of nr_samples taken
     figure;
     plot(nr_samples_list, listAvgRMS);
     title(['Elbow' , ', selectionType = ', num2str(selectionType)])
     xlabel('sample size')
     ylabel('RMS')
     axis([0 1100 0.15 0.23])
     
     % Plotting the RMS as a function of iterations completed
     % only consider last nr_samples value
     figure
     plot(listRMS, '-o')
     title(['RMS per iteration', ', selectionType = ', num2str(selectionType) ])
     xlabel('step')
     ylabel('RMS')  

   %break;
end
