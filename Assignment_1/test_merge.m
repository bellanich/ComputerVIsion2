% Testing ICP on accuracy, stability hyperparameters etc.
%% Initialize file to store data
% Only run if you want to overwrite previously existing file
fileID = fopen('merge_testing.txt','w');

%% Run ICP for particular selectionType, and particular nr_samples

clear all
clc
close all

sampling_rates = [1, 2, 4, 10, 1];


for i = 1:5
    if i < 5
        method = 'separated_merge';
    else
        method = 'combined_merge';
    end
    [cloud, RMS, time] = merge(method, sampling_rates(i));
    
    % save in .txt file
    fileID = fopen('merge_testing.txt','a');
    result_values = ['RMS = ', num2str(RMS), ', time = ', num2str(time)];
    results = [result_values, newline];
    fprintf(fileID,'%10s \n', results);
    fclose(fileID);
end
