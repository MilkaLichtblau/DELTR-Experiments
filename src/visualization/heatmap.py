'''
Created on Apr 11, 2018

@author: meike.zehlike

creates a heatmap from the Chile University dataset to show correlations between highschool grades
as well as university entrance tests and the success in university after a year

also asks if the success after one year is gender specific or schooling specific (public vs semi-
public vs private schools)
'''

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import data_preparation.chileDatasetPreparation as prepChile

#################################################################################################
# HEATMAPS
#################################################################################################

def cool_warm_heatmap(data, filename):
    corr = data.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
        linewidths=.5, annot_kws={"size":8})
    fig = hm.get_figure()
    fig.savefig(filename, pad_inches=1, bbox_inches='tight')

# plot_ChileDataset heatmap for successful students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prepChile.principalDataPreparation_withSemiPrivate(data)
data = prepChile.successfulStudents(data)
cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapSuccessfulStudents.png')

# plot_ChileDataset heatmap for all students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prepChile.principalDataPreparation_withSemiPrivate(data)
data = prepChile.allStudents(data)
cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapAllStudents.png')

# plot Law Student heatmaps for gender and race
training_data = pd.read_csv('../../octave-src/sample/LawStudents/gender/LawStudents_Gender_train.txt',
                            sep=",", names=['query', 'sex', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/gender/heatmapGender.png')

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'asian', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/race_asian/heatmapAsian.png')

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_black/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'black', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/race_black/heatmapBlack.png')

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'hispanic', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/race_hispanic/heatmapHispanic.png')

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'mexican', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/race_mexican/heatmapMexican.png')

training_data = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train.txt',
                            sep=",", names=['query', 'puertorican', 'LSAT', 'UGPA', 'ZFYA'])
cool_warm_heatmap(training_data, '../../octave-src/sample/LawStudents/race_puertorican/heatmapPuertorican.png')
