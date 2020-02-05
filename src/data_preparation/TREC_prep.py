'''
Created on Aug 7, 2018

@author: mzehlike
'''

import pandas as pd


def prepare():
    rootPath = '../data/TREC/'
    data = pd.read_csv(rootPath + 'features_withListNetFormat_withGender_withZscore_candidateAmount-200_total.csv',
                       sep=',',
                       names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])

    test_queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_fold1 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold1 = data.loc[data['query_id'].isin(test_queries)]

    train_fold1.to_csv(rootPath + 'fold_1/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold1.to_csv(rootPath + 'fold_1/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    test_queries = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    train_fold2 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold2 = data.loc[data['query_id'].isin(test_queries)]

    train_fold2.to_csv(rootPath + 'fold_2/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold2.to_csv(rootPath + 'fold_2/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    test_queries = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    train_fold3 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold3 = data.loc[data['query_id'].isin(test_queries)]

    train_fold3.to_csv(rootPath + 'fold_3/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold3.to_csv(rootPath + 'fold_3/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    test_queries = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    train_fold4 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold4 = data.loc[data['query_id'].isin(test_queries)]

    train_fold4.to_csv(rootPath + 'fold_4/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold4.to_csv(rootPath + 'fold_4/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    test_queries = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    train_fold5 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold5 = data.loc[data['query_id'].isin(test_queries)]

    train_fold5.to_csv(rootPath + 'fold_5/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold5.to_csv(rootPath + 'fold_5/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    test_queries = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    train_fold6 = data.loc[~data['query_id'].isin(test_queries)]
    test_fold6 = data.loc[data['query_id'].isin(test_queries)]

    train_fold6.to_csv(rootPath + 'fold_6/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
    test_fold6.to_csv(rootPath + 'fold_6/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

    print("created 6-folds trec dataset with " + str(data['gender'].value_counts()[0]) + \
              " men and " + str(data['gender'].value_counts()[1]) + " women.")

