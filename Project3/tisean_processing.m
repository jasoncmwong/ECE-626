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

% Get dimensionality of feature vectors for each dataset
glass_m = glass_d * glass_tau;
laser_m = laser_d * laser_tau;

% Convert datasets into (delay vectors, next observation)
[glass_inputs, glass_targets] = delay_embed(glass, glass_m, glass_tau);
[laser_inputs, laser_targets] = delay_embed(laser, laser_m, laser_tau);

% Write delay embedded datasets into .csv files
csvwrite('glass_delay.csv', [glass_inputs glass_targets]);
csvwrite('laser_delay.csv', [laser_inputs laser_targets]);

%===FUNCTIONS===%
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
end

% Check that a local minimum was reached
if i == size(mi_info, 1)
    disp('Local minimum never reached\n');
end
end

% Obtain time-series dataset for neural network input by converting the
% data into delay vectors and next observations
function [inputs, targets] = delay_embed(time_series, m, tau)
max_time = length(time_series);

% Initialize input indices and target index, along with arrays to hold
% inptus and targets
input_ind = 1:tau:tau*m;
target_ind = tau*(m+1);

for i = 1:(max_time-target_ind+1)
    inputs(i, :) = time_series(input_ind);
    targets(i) = time_series(target_ind);
    input_ind = input_ind + 1;
    target_ind = target_ind + 1;
end
targets = targets';
end
