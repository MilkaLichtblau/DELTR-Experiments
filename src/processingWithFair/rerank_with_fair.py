import processingWithFair.fair.post_processing_methods.fair_ranker.create as fair
from processingWithFair.fair.dataset_creator.candidate import Candidate

import pandas as pd


def rerank_featurevectors(dataDescription, p_deviation=0.0, post_process=False):
    data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header)
    data['uuid'] = 'empty'
    reranked_features = pd.DataFrame()

    # re-rank with fair for every query
    for query in data['query_id'].unique():
        # print("Rerank for query " + str(query))
        data_query = data.query("query_id==" + str(query))
        data_query, protected, nonProtected = create(data_query, dataDescription)

        # protected attribute value is always 1
        p = (len(data_query.query(dataDescription.protected_attribute + "==1")) / len(data_query) + p_deviation)

        if("TREC" in dataDescription.orig_data_path):
            p = 0.105 + p_deviation

        fairRanking, _ = fair.fairRanking(data_query.shape[0], protected, nonProtected, p, dataDescription.alpha)
        fairRanking = setNewQualifications(fairRanking)

        # swap original qualification with fair qualification
        for candidate in fairRanking:
            candidate_row = data_query[data_query.uuid == candidate.uuid]

            candidate_row.iloc[0, data_query.columns.get_loc(dataDescription.judgment)] = (candidate.qualification / len(fairRanking))

            reranked_features = reranked_features.append(candidate_row.iloc[0], sort=False)
            reranked_features = reranked_features[candidate_row.columns]

    # sort by judgment to ease evaluation of output
    reranked_features_sorted = pd.DataFrame()
    for query in data['query_id'].unique():
        sortet = reranked_features.query("query_id==" + str(query)).sort_values(by=dataDescription.judgment, ascending=False)
        reranked_features_sorted = reranked_features_sorted.append(sortet)

    reranked_features_sorted.update(reranked_features_sorted['query_id'].astype(int).astype(str))

    # bring file into expected format for evaluation, if used for post-processing
    if post_process:
        reranked_features_sorted = reranked_features_sorted.drop(columns=['uuid'])
        reranked_features_sorted = reranked_features_sorted.astype({'prot_attr' : 'int64', 'rank' : 'int64'})
        reranked_features_sorted = reranked_features_sorted[dataDescription.header]
    else:
        reranked_features_sorted = reranked_features_sorted.drop(columns=['uuid'])
    reranked_features_sorted.to_csv(dataDescription.result_path, sep=',', index=False, header=False)


def create(data, dataDescription):
    protected = []
    nonProtected = []

    for row in data.itertuples():
        # change to different index in row[.] to access other columns from csv file
        if row[data.columns.get_loc(dataDescription.protected_attribute) + 1] == 0.:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], [])
            nonProtected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid

        else:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], dataDescription.protected_group)
            protected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid

    # sort candidates by judgment
    protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
    nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)

    return data, protected, nonProtected


def setNewQualifications(fairRanking):
    qualification = len(fairRanking)
    for candidate in fairRanking:
        candidate.qualification = qualification
        qualification -= 1
    return fairRanking

