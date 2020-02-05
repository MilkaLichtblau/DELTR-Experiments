#!/usr/bin/octave -qf
% train a linear network using the examples provided in the training_file
% and writes the model on model_file

% suppress output
more off;

% pararrayfun is in the 'general' package
pkg load general;
pkg load parallel;

% read arguments on the command line
arg_list = argv();

result_directory = arg_list{1,1}
training_file = arg_list{2,1}
model_file = arg_list{3,1}
GAMMA = str2num(arg_list{4,1})

% load constants
addpath(".")
source "./globals.m";

% load training dataset
disp('loading training data...')
data = load(training_file);
list_id = data(:,1);
X = data(:,2:size(data,2)-1);
y = data(:,size(data,2));

% launch the training routine
disp(sprintf('training, %d iteration, %d examples, learning rate %f, gamma %d, ...\n', T, size(X,1), e, GAMMA))
tic();
omega = trainNN(GAMMA, result_directory, list_id, X, y, T, e);
training_time = toc();
disp(sprintf('finished training, time elapsed: %d seconds', training_time))
save(model_file, "omega");
