#!/usr/bin/env octave
% command line arguments:
% predict.m model feature_file

% suppress output
more off;


% load constants
addpath(".")
source "./globals.m";

arg_list = argv();
test_file = arg_list{1,1}
model_file = arg_list{2,1}
output_dir = arg_list{3,1}

% if this model is colorblind, exclude the protected feature from prediction
if isempty(strfind(model_file, "COLORBLIND"))
	FEAT_START = 2
else
	FEAT_START = 3
endif

omega = load(model_file);
drg = load(test_file);

list_id = drg(:,1);
X = drg(:,FEAT_START:size(drg,2)-FEAT_END);

omega_values = omega.omega(:);

z =  X * omega_values;
doc_ids = 1:size(z);

# also write y for later evaluation
y = drg(:, size(drg,2));

if isempty(strfind(model_file, "COLORBLIND"))
	# write protected attributed, if this is not a colorblind prediction
	y = [list_id, doc_ids', y, drg(:, FEAT_START)];
else
	y = [list_id, doc_ids', y];
endif

filename = [output_dir "trainingScores_ORIG.pred"];
dlmwrite(filename, y);

# add document ids for later evaluation
if isempty(strfind(model_file, "COLORBLIND"))
	# write protected attributed, if this is not a colorblind prediction
	z = [list_id, doc_ids', z, drg(:, FEAT_START)];
else
	z = [list_id, doc_ids', z];
endif

# add a little random to avoid ties
r = @(i) (i+rand*0.02-0.01);

for id = unique(list_id)'
    indexes = find(list_id==id);
    z_temp = z(indexes, :);
    z(indexes, :) = sortrows(z_temp, -3);
endfor
sorted_ranks = z;
filename = [output_dir "predictions.pred"];

dlmwrite(filename, sorted_ranks)

