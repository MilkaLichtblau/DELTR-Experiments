#!/usr/bin/octave -qf

% suppress output
more off;


% load constants
addpath(".")
source "./globals.m";

arg_list = argv();

test_file = arg_list{1,1}
model_file = arg_list{2,1}
output_dir = arg_list{3,1}

omega = load(model_file);

drg = load(test_file);

list_id = drg(:,1);
X = drg(:,2:size(drg,2)-1);

z =  X * omega.omega;
doc_ids = 1:size(z);

# also write y for later evaluation
y = drg(:, size(drg,2));
y = [list_id, doc_ids', y, X(:, PROT_COL)];

filename = [output_dir "trainingScores_ORIG.pred"];
dlmwrite(filename, y);


# add protection status to a for later evaluation
z = [z, X(:, PROT_COL)];

# add list ids and document ids for later evaluation
z = [list_id, doc_ids', z];

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
%figure(); plot(z);
