import processingWithFair.rerank_with_fair as rerank
from processingWithFair.DatasetDescription import DatasetDescription


class Preprocessing():

    def __init__(self, dataset, p):
        self.dataset = dataset
        if p == "p_minus":
            self.p = -0.1
            self.description_classifier = "RERANKED_PMinus"
        elif p == "p_plus":
            self.p = 0.1
            self.description_classifier = "RERANKED_PPlus"
        else:
            self.p = 0.0
            self.description_classifier = "RERANKED"

    def preprocess_dataset(self):

        if self.dataset == "trec":
            """
            TREC Data

            """
            print("Start reranking of TREC Data")
            protected_attribute = "gender"
            protected_group = "female"
            header = ["query_id", "gender", "match_body_email_subject_score_norm", "match_body_email_subject_df_stdev",
                      "match_body_email_subject_idf_stdev", "match_body_score_norm", "match_subject_score_norm", "judgment"]
            judgment = "judgment"

            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6"]:
                print("Reranking for " + fold)
                origFile = "../data/TREC/" + fold + "/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv"
                resultFile = "../data/TREC/" + fold + "/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_" + self.description_classifier + ".csv"
                TRECData = DatasetDescription(resultFile,
                                              origFile,
                                              protected_attribute,
                                              protected_group,
                                              header,
                                              judgment)

                rerank.rerank_featurevectors(TRECData, self.p)

        if self.dataset == "law":
            """
            LSAT Data - Gender

            """
            print("Start reranking of LSAT Data")
            print("protected attribute: sex")

            origFile = "../data/LawStudents/gender/LawStudents_Gender_train.txt"
            resultFile = "../data/LawStudents/gender/LawStudents_Gender_train_" + self.description_classifier + ".txt"
            protected_attribute = "sex"
            protected_group = "female"
            header = ["query_id", "sex", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"

            LSATGenderData = DatasetDescription(resultFile,
                                                origFile,
                                                protected_attribute,
                                                protected_group,
                                                header,
                                                judgment)

            rerank.rerank_featurevectors(LSATGenderData, self.p)

            """
            LSAT Data - Race - Black

            """
            if self.p < 0:
                # the black group has only 6% blacks, so p_minus is not possible
                return
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: black")

            resultFile = "../data/LawStudents/race_black/LawStudents_Race_train_" + self.description_classifier + ".txt"
            origFile = "../data/LawStudents/race_black/LawStudents_Race_train.txt"
            protected_attribute = "race"
            protected_group = "black"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"

            LSATRaceBlackData = DatasetDescription(resultFile,
                                                   origFile,
                                                   protected_attribute,
                                                   protected_group,
                                                   header,
                                                   judgment)

            rerank.rerank_featurevectors(LSATRaceBlackData, self.p)

        if self.dataset == "engineering-NoSemi":

            """
            Engineering Students Data - NoSemi - gender

            """
            print("Start reranking of Engineering Students Data - No Semi Private - gender")
            protected_attribute = "hombre"
            protected_group = "female"
            header = ['query', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                origFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                rerank.rerank_featurevectors(EngineeringData, self.p)

                fold_count += 1

            """
            Engineering Students Data - NoSemi - highschool

            """
            print("Start reranking of Engineering Students Data - No Semi Private - highschool")
            protected_attribute = "highschool_type"
            protected_group = "highschool"
            header = ['query', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                origFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                rerank.rerank_featurevectors(EngineeringData, self.p)

                fold_count += 1

