'''
Created on Jul 24, 2018

@author: mzehlike
'''

import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

import pandas as pd


def determineGroups(attributeNamesAndCategories):
    elementSets = []
    groups = []
    for attr, cardinality in attributeNamesAndCategories.items():
        elementSets.append(list(range(0, cardinality)))

    groups = list(itertools.product(*elementSets))
    return groups


def separate_groups(data_set, categories, attributeItems):
    num_categories = len(categories)
    separateByGroups = [[] for _ in range(num_categories)]

    for idx, row in data_set.iterrows():
        categorieList = []
        for j in attributeItems:
            col_name = j[0]
            categorieList.append(row[col_name])
        separateByGroups[categories.index(tuple(categorieList))].append(row)
        categorieList = []
    return separateByGroups


def plot_LSATDataset(data_set, attributeNamesAndCategories, attributeQuality, filename, labels):

    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True


    colors = ['blue', 'red', 'green']
    markers = ['-', '--', '-.', '-+', '-d']
    data = data_set.sort_values(by=attributeQuality, ascending=False)
    best = data[attributeQuality].iloc[0]
    categories = determineGroups(attributeNamesAndCategories)
    attributeItems = attributeNamesAndCategories.items()
    output_ranking_separated = separate_groups(data, categories, attributeItems)
    separateQualityByGroups = []
    fig = plt.figure(figsize=(12, 7))
    round_2f = []
    for idx, row in data.iterrows():
        row[attributeQuality] = float(row[attributeQuality]) / best

    for i in range(len(categories)):
        separateQualityByGroups.append([quality[attributeQuality] for quality in output_ranking_separated[i]])
        fit = stats.norm.pdf(separateQualityByGroups[i], np.mean(separateQualityByGroups[i]), np.std(separateQualityByGroups[i]))
        plt.plot(separateQualityByGroups[i], fit, markers[i], markersize=6, label=labels[i], color=colors[i])
        round_2f.append([round(elem, 2) for elem in separateQualityByGroups[i]])

    # erase underscores as last characters
    attributeQuality = attributeQuality.replace('_', '\_')

    plt.xlabel(attributeQuality)
    plt.legend(loc='upper left')
    plt.savefig(filename, dpi=100, bbox_inches='tight')



training_data = pd.read_csv('../../octave-src/sample/LawStudents/gender/LawStudents_Gender_train.txt',
                            sep=",", names=['query', 'sex', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"sex" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/gender/zfya-distribution.png',
                 labels=['male', 'female'])

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'race', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"race" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/race_asian/zfya-distribution.png',
                 labels=['white', 'asian'])

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_black/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'race', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"race" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/race_black/zfya-distribution.png',
                 labels=['white', 'black'])

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'race', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"race" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/race_hispanic/zfya-distribution.png',
                 labels=['white', 'hispanic'])

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'race', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"race" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/race_mexican/zfya-distribution.png',
                 labels=['white', 'mexican'])

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'race', 'LSAT', 'UGPA', 'ZFYA'])
attributeNamesAndCategories = {"race" : 2}
attributeQuality = "ZFYA"
plot_LSATDataset(training_data, attributeNamesAndCategories, attributeQuality,
                 '../../octave-src/sample/LawStudents/race_puertorican/zfya-distribution.png',
                 labels=['white', 'puertorican'])



