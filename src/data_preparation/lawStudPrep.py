'''
Created on May 28, 2018

@author: meike.zehlike

protected attributes: sex, race
features: Law School Admission Test (LSAT), grade point average (UGPA)

training judgments: first year average grade (ZFYA)

excluding for now: region-first, sander-index, first_pf


höchste ID: 27476

Aufteilung in Trainings und Testdaten, 80% Training, 20% Testing, Random Sampling
'''

import pandas as pd
from scipy.stats import stats


def prepareGenderData():
    data_root_path = "../data/LawStudents/"
    data = pd.read_excel(data_root_path + 'law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'race'])

    data['sex'] = data['sex'].replace([2], 0)

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['sex', 'LSAT', 'UGPA', 'ZFYA']]
    data.insert(0, 'query\_dummy', 1)

    train = data.sample(frac=.8)
    test = data.drop(train.index)

    # subsample training data, because otherwise training takes too long
    train = train.sample(frac=.1)
    train = train.sort_values(by=['ZFYA'], ascending=False)
    test = test.sort_values(by=['ZFYA'], ascending=False)

    print("created law students dataset with " + str(data['sex'].value_counts()[0]) + \
              " men and " + str(data['sex'].value_counts()[1]) + " women.")

    return train, test


def prepareRaceData(protGroup, nonprotGroup):
    data_root_path = "../data/LawStudents/"
    data = pd.read_excel(data_root_path + 'law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

    data['race'] = data['race'].replace(to_replace=protGroup, value=1)
    data['race'] = data['race'].replace(to_replace=nonprotGroup, value=0)

    data = data[data['race'].isin([0, 1])]

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['race', 'LSAT', 'UGPA', 'ZFYA']]
    data.insert(0, 'query\_dummy', 1)

    data = data.sort_values(by=['ZFYA'], ascending=False)

    train = data.sample(frac=.8)
    test = data.drop(train.index)

    # subsample training data, because otherwise training takes too long
    train = train.sample(frac=.1)
    train = train.sort_values(by=['ZFYA'], ascending=False)
    test = test.sort_values(by=['ZFYA'], ascending=False)

    print("created law students dataset with " + str(data['race'].value_counts()[0]) + " " + nonprotGroup + \
              " people and " + str(data['race'].value_counts()[1]) + " " + protGroup + " people.")

    return train, test


def prepareAllInOneDataForFAIR():
    data = pd.read_excel(data_root_path + 'sample/LawStudents/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf'])

    data['sex'] = data['sex'].replace([2], 0)

    data['race'] = data['race'].replace(to_replace="White", value=0)
    data['race'] = data['race'].replace(to_replace="Amerindian", value=1)
    data['race'] = data['race'].replace(to_replace="Asian", value=2)
    data['race'] = data['race'].replace(to_replace="Black", value=3)
    data['race'] = data['race'].replace(to_replace="Hispanic", value=4)
    data['race'] = data['race'].replace(to_replace="Mexican", value=5)
    data['race'] = data['race'].replace(to_replace="Other", value=6)
    data['race'] = data['race'].replace(to_replace="Puertorican", value=7)

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['sex', 'race', 'LSAT', 'UGPA', 'ZFYA']]

    return data

