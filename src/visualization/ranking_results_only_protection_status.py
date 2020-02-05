'''
Created on Apr 26, 2018

@author: meike.zehlike
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PROT_COL_PRED = 3
PROT_COL_ORIG = 1
PROT_ATTR = 1


def plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, k, step_size):
    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True

    f, ax = plt.subplots(figsize=(20, 3))

    # Generate dummy info for plot_ChileDataset handles "h"
#     red_circle, = plt.plot(0, 1.65, 'ro', ms=14, label="protected")
#     blue_plus, = plt.plot(0, 1.65, 'b+', ms=16, mew=3, label="non-protected");
    markersize_bullet = 11
    markersize_plus = 13
    data = pd.read_csv(input_file1, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL_ORIG] == PROT_ATTR:
            plt.plot(i, 1, 'ro', ms=markersize_bullet);
        else:
            plt.plot(i, 1, 'b+', ms=markersize_plus, mew=3);

    data = pd.read_csv(input_file2, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL_PRED] == PROT_ATTR:
            plt.plot(i, 1.25, 'ro', ms=markersize_bullet);
        else:
            plt.plot(i, 1.25, 'b+', ms=markersize_plus, mew=3);

    data = pd.read_csv(input_file3, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL_PRED] == PROT_ATTR:
            plt.plot(i, 1.5, 'ro', ms=markersize_bullet);
        else:
            plt.plot(i, 1.5, 'b+', ms=markersize_plus, mew=3);

    data = pd.read_csv(input_file4, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL_PRED] == PROT_ATTR:
            plt.plot(i, 1.75, 'ro', ms=markersize_bullet);
        else:
            plt.plot(i, 1.75, 'b+', ms=markersize_plus, mew=3);

    plt.gca().invert_yaxis()
    plt.xlabel ("ranking position");
    plt.yticks(np.arange(1, 2, step=.25), ('Ground Truth', 'Normal L2R', 'Small Gamma', 'Large Gamma'))
    # plt.legend([red_circle, blue_plus], ['protected', 'non-protected'], loc='upper right')  # , bbox_to_anchor=(0.73, 0.92))

    plt.savefig(output_file, bbox_inches='tight')

