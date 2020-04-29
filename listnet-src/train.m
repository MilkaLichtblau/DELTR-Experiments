#!/usr/bin/octave -qf
% train a linear network using the examples provided in the training_set file
% and writes the model on output_model
%
% usage: # train.m training_set output_model

% suppress output
more off;

% pararrayfun is in the 'general' package
pkg load general;
pkg load parallel;

% read arguments on the command line
arg_list = argv ();


directory = arg_list{1,1}
training_file = arg_list{2,1}
model_file = arg_list{3,1}

% if this experiment is colorblind, exclude the protected feature from training
if isempty(strfind(directory, "COLORBLIND"))
	FEAT_START = 2
else
	FEAT_START = 3
endif

% if this experiment is pre-processing, exclude the last feature from training which is the document uuid
%if isempty(strfind(training_file, "RERANKED"))
%	FEAT_END = 1
%else
%	FEAT_END = 2
%endif

% load constants
addpath(".")
source "./globals.m";

% load training dataset
disp('loading training data...')
data = csvread(training_file);
list_id = data(:,1);
X = data(:,FEAT_START:size(data,2)-1);
y = data(:,size(data,2));

% launch the training routine
disp(sprintf('training, %d iteration, %d examples, learning rate %f...', T, size(X,1), e))
tic();
omega = trainNN(list_id, directory, X, y, T, e);
training_time = toc();
disp(sprintf('finished training, time elapsed: %d seconds', training_time))
save(model_file, "omega");
