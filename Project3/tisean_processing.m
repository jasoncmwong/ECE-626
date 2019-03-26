% TISESAN pre-processing
clear; clc;

% Set path to TISEAN binary executables
tisean_path = 'C:\Users\Jason\Dropbox\University\Tisean\bin\';

% Load data
glass = load('mackey-glass.dat');
laser = load('santa-fe-laser-a.dat');

% Find mutual information for datasets using TISEAN mutual command
glass_mi_cmd = sprintf('mutual -o glass_mi.dat mackey-glass.dat');
laser_mi_cmd = sprintf('mutual -o laser_mi.dat santa-fe-laser-a.dat');

system([tisean_path, glass_mi_cmd]);
glass_mi = dlmread('glass_mi.dat', ' ', 1, 0);

system([tisean_path, laser_mi_cmd]);
laser_mi = dlmread('laser_mi.dat', ' ', 1, 0);

% Get best tau for both datasets
glass_tau = get_tau(glass_mi);
laser_tau = get_tau(laser_mi);

% Find embedding dimension for datasets using TISEAN false_nearest command
glass_nn_cmd = sprintf('false_nearest -d%u -M1,20 -f12.5 -o glass_nn.dat mackey-glass.dat', glass_tau);
laser_nn_cmd = sprintf('false_nearest -d%u -M1,20 -f12.5 -o laser_nn.dat santa-fe-laser-a.dat', laser_tau);

system([tisean_path, glass_nn_cmd]);
glass_nn = load('glass_nn.dat');

system([tisean_path, laser_nn_cmd]);
laser_nn = load('laser_nn.dat');

% Get minimum embedding dimension for both datasets
FNN_THRESH = 0;
[f_fnn, glass_d] = min(glass_nn(:, 2));
if (f_fnn > FNN_THRESH)
    disp('Fraction of false nearest neighbours did not meet threshold (glass)\n');
end
[f_fnn, laser_d] = min(laser_nn(:, 2));

if (f_fnn > FNN_THRESH)
    disp('Fraction of false nearest neighbours did not meet threshold (laser)\n');
end

% Determines optimal time delay based on finding the first local minimum of
% the mutual information between a sample and its sample t steps after
function tau = get_tau(mi_info)
% Iterate through the mutual information results and find first local min
min_val = inf;
tau = 0;

for i = 1:size(mi_info, 1)
    if (mi_info(i, 2) < min_val)
        min_val = mi_info(i, 2);
        tau = i - 1;
    else
        break;  % Mutual information is starting to increase
    end
    disp('Local minimum never reached\n');
end
end
