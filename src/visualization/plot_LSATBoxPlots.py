'''
Created on Aug 15, 2018

@author: mzehlike
'''

'''
Created on Apr 17, 2018

@author: mzehlike
'''

import data_preparation.lawStudentDatasetPreparation as prep
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################################
# BOXPLOTS
##############################################################################################

def boxPlot(data, filename, prot_attr, gender=True):
    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True

    # Grouped boxplot
    fig, ax = plt.subplots(figsize=(20, 10))

    df_melt = data.melt(id_vars=prot_attr,
                  value_vars=['LSAT',
                                'UGPA',
                                'ZFYA'],
                  var_name='columns')

    my_pal = {0: "b", 1: "orangered"}
    ax = sns.boxplot(data=df_melt,
                hue=prot_attr,  # different colors for different 'cls'
                x='columns',
                y='value',
                showfliers=False,
                palette=my_pal)



    labels = ['LSAT', 'Highschool\nGrades', 'University\nGrades']
    ax.set_xticklabels(labels)
    handles, labels = ax.get_legend_handles_labels()
    if (gender):
        ax.legend(loc='upper left', handles=handles, labels=['Male', 'Female'])
        plt.title('Law Student SAT Scores by Gender')
    else:
        ax.legend(loc='upper left', handles=handles, labels=['White', 'Black'])
        plt.title('Law Student SAT Scores by Ethnicity')

    plt.xlabel("")
    plt.ylabel("z-scores")
    plt.savefig(filename, bbox_inches='tight')

train, test = prep.prepareGenderData()
data = pd.concat([train, test])
boxPlot(data, '../../octave-src/sample/LawStudents/gender/boxPlots_LSAT_Gender.png', 'sex')

train, test = prep.prepareRaceData('Black', 'White')
data = pd.concat([train, test])
boxPlot(data, '../../octave-src/sample/LawStudents/race_black/boxPlots_LSAT_Race.png', 'race', gender=False)





