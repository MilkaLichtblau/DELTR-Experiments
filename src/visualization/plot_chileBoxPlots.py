'''
Created on Apr 17, 2018

@author: mzehlike
'''

import data_preparation.chileDatasetPreparation as prep
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################################
# BOXPLOTS
##############################################################################################

def boxPlot(data, filename, gender=True):
    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True

    # Grouped boxplot
    fig, ax = plt.subplots(figsize=(20, 10))

    df_melt = data.melt(id_vars='prot\_attr',
                  value_vars=['psu\_mat',
                                'psu\_len',
                                'psu\_cie',
                                'nem',
                                'score'],
                  var_name='columns')

    my_pal = {0: "b", 1: "orangered"}
    ax = sns.boxplot(data=df_melt,
                hue='prot\_attr',  # different colors for different 'cls'
                x='columns',
                y='value',
                showfliers=False,
                palette=my_pal)



    labels = ['PSU Math', 'PSU Language', 'PSU Science', 'Highschool\nGrades', 'University\nGrades']
    ax.set_xticklabels(labels)
    handles, labels = ax.get_legend_handles_labels()
    if (gender):
        ax.legend(loc='upper left', handles=handles, labels=['Male', 'Female'])
        plt.title('Chile University Scores by Gender')
    else:
        ax.legend(loc='upper left', handles=handles, labels=['Private', 'Public'])
        plt.title('Chile University Scores by Highschool Type')

    plt.xlabel("")
    plt.ylabel("z-scores")
    plt.savefig(filename, bbox_inches='tight')



#############################################################################################
# with semi-private
#############################################################################################

# plot for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.prepareForBoxplots(data, gender=True)
boxPlot(data, '../../plots/Chile/boxPlots_Gender_semi.png')


# plot for highschool
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withSemiPrivate(data)
data = prep.prepareForBoxplots(data, gender=False)
boxPlot(data, '../../plots/Chile/boxPlots_Highschool_semi.png', gender=False)

#############################################################################################
# without semi-private
#############################################################################################

# plot for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withoutSemiPrivate(data)
data = prep.prepareForBoxplots(data, gender=True)
boxPlot(data, '../../plots/Chile/boxPlots_Gender_nosemi.png')


# plot for highschool
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withoutSemiPrivate(data)
data = prep.prepareForBoxplots(data, gender=False)
boxPlot(data, '../../plots/Chile/boxPlots_Highschool_nosemi.png', gender=False)








