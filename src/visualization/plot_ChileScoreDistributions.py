'''
Created on Apr 17, 2018

@author: mzehlike
'''

import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

import pandas as pd
import data_preparation.chileDatasetPreparation as prep


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


def plot_ChileDataset(data_set, attributeNamesAndCategories, attributeQuality, filename, labels):

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


############################################################################
# UNIVERSITY GRADES AFTER FIRST YEAR
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "notas_"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_SuccessfulStudents.png',
         labels=['male', 'female'])

############################################################################
# HIGHSCHOOL FINAL GRADES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "nem"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_SuccessfulStudents.png',
         labels=['male', 'female'])


############################################################################
# PSU MATH SCORES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "psu_mat"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_SuccessfulStudents.png',
         labels=['male', 'female'])


############################################################################
# PSU LANGUAGE SCORES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "psu_len"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSULang_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSULang_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSULang_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSULang_SuccessfulStudents.png',
         labels=['male', 'female'])

############################################################################
# Fraction of Credits a Student failed
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "rat_ud"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.allStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.successfulStudents(data)
plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_SuccessfulStudents.png',
         labels=['male', 'female'])




def plot_syntheticDataset(data_set, attributeNamesAndCategories, attributeQuality, filename, labels, normal=False):

    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
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
        if normal:
            fit = stats.norm.pdf(separateQualityByGroups[i], np.mean(separateQualityByGroups[i]), np.std(separateQualityByGroups[i]))
        else:
            fit = stats.uniform.pdf(separateQualityByGroups[i], np.mean(separateQualityByGroups[i]), np.std(separateQualityByGroups[i]))
        plt.plot(separateQualityByGroups[i], fit, markers[i], markersize=6, label=labels[i], color=colors[i])
        round_2f.append([round(elem, 2) for elem in separateQualityByGroups[i]])

    plt.xlabel(attributeQuality)
    plt.legend(loc='upper left')
    plt.savefig(filename, dpi=100, bbox_inches='tight')



