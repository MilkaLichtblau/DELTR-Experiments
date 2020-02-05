'''
Created on May 2, 2018

@author: mzehlike
'''
import pandas as pd
import visualization.ranking_results_only_protection_status as rankres

attributeNamesAndCategories = {"gender" : 2}
attributeQuality = "score"

###########################################################################
# UNIFORM TOP MALE BOTTOM FEMALE
###########################################################################

# DATASET PLOT
# data = pd.read_csv('../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_train_data_scoreAndGender_separated.txt',
#                    sep=",", names=["query_id", "gender", "score", "rank"])
#
# pdf.plot_syntheticDataset(data, attributeNamesAndCategories, attributeQuality,
#          '../../plots/synthetic/separated/top_male_bottom_female/dataset_plots/uniform_distribution/uniform_male_top_pdf_scores.png',
#          labels=['non-protected', 'protected'])

# RESULT PLOT
input_file1 = '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt'
input_file2 = '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/predictions_SORTED.pred'
input_file3 = '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/predictions_SORTED.pred'
input_file4 = '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/predictions_SORTED.pred'
output_file = '../../plots/synthetic/separated/top_male_bottom_female/result_plots/uniform_distribution/uniform_male_top_rankings_group_property_only.png'

rankres.plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, 50, 1)

###########################################################################
# UNIFORM TOP FEMALE BOTTOM MALE
###########################################################################

# DATASET PLOT
# data = pd.read_csv('../../octave-src/sample/synthetic_score_gender/top_female_bottom_male/sample_train_data_scoreAndGender_separated.txt',
#                    sep=",", names=["query_id", "gender", "score", "rank"])
#
# pdf.plot_syntheticDataset(data, attributeNamesAndCategories, attributeQuality,
#          '../../plots/synthetic/separated/top_female_bottom_male/dataset_plots/uniform_distribution/uniform_female_top_pdf_scores.png',
#          labels=['non-protected', 'protected'])
#
# RESULT PLOT
# input_file1 = '../../octave-src/sample/synthetic/top_female_bottom_male/sample_test_data_scoreAndGender_separated.txt'
# input_file2 = '../../octave-src/sample/synthetic/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_ZERO.txt.pred'
# input_file3 = '../../octave-src/sample/synthetic/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_MEDIUM.txt.pred'
# input_file4 = '../../octave-src/sample/synthetic/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_LARGE.txt.pred'
# output_file = '../../plots/synthetic/separated/top_female_bottom_male/result_plots/uniform_distribution/uniform_female_top_rankings_group_property_only.png'
#
# rankres.plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, 50, 1)

