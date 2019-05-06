function successful = install_dependencies()

%{
    Download the following libraries into the give directories:
    - liblinear   -> <project_root>/modules/liblinear
    - vlfeat      -> <project_root>/modules/vlfeat-0.9-21
    - matconvnet  -> <project_root>/modules/matconvnet-master
%}


project_root = pwd;
mex -setup C++;

% ##################
%   Install vlfeat
% ##################
run(project_root + "/modules/vlfeat-0.9.21/toolbox/vl_setup");

% ##################
% Install matconvnet 
% ##################
run(project_root + "/modules/matconvnet-master/matlab/vl_setupnn");

% ##################
% Install liblinear 
% ##################
addpath modules/liblinear;
cd modules/liblinear/;
make;
cd ..;
cd ..;

successful = true;
