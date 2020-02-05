import processingWithFair.rerank_with_fair as rerank
import pandas as pd
from processingWithFair.DatasetDescription import DatasetDescription


class Postprocessing():

    def __init__(self, dataset, p):
        self.dataset = dataset
        if p == "p_minus":
            self.p = -0.1
            self.p_classifier = "P-Minus"
        elif p == "p_plus":
            self.p = 0.1
            self.p_classifier = "P-Plus"
        else:
            self.p = 0.0
            self.p_classifier = "P-Star"

    def postprocess_result (self):
        header = ["query_id", "rank", "pred_score", "prot_attr"]
        protected_attribute = "prot_attr"
        judgment = "pred_score"
        if self.dataset == "trec":
            """
            TREC Data
            """
            print("Start post-processing TREC Data")
            protected_group = "female"

            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6"]:
                print("Post-processing for " + fold + " with " + self.p_classifier)
                origPredictions = "../results/TREC/" + fold + "/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
                rerankedPredictions = "../results/TREC/" + fold + "/FA-IR/" + self.p_classifier + "/predictions.pred"
                TRECData = DatasetDescription(rerankedPredictions,
                                              origPredictions,
                                              protected_attribute,
                                              protected_group,
                                              header,
                                              judgment)

                rerank.rerank_featurevectors(TRECData, self.p, post_process=True)

        if self.dataset == "law":
            """
            LSAT Data - Gender

            """
            print("Start post-processing of LSAT Data" + " with " + self.p_classifier)
            print("protected attribute: sex")

            origPredictions = "../results/LawStudents/gender/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
            rerankedPredictions = "../results/LawStudents/gender/FA-IR/" + self.p_classifier + "/predictions.pred"
            protected_group = "female"

            LSATGenderData = DatasetDescription(rerankedPredictions,
                                                origPredictions,
                                                protected_attribute,
                                                protected_group,
                                                header,
                                                judgment)

            rerank.rerank_featurevectors(LSATGenderData, self.p, post_process=True)

            """
            LSAT Data - Race - Black

            """
            if self.p < 0:
                # the black group has only 6% blacks, so p_minus is not possible
                return
            print("Start post-processing of LSAT Data" + " with " + self.p_classifier)
            print("protected attribute: race - protected group: black")

            origPredictions = "../results/LawStudents/race_black/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
            rerankedPredictions = "../results/LawStudents/race_black/FA-IR/" + self.p_classifier + "/predictions.pred"
            protected_group = "black"

            LSATRaceBlackData = DatasetDescription(rerankedPredictions,
                                                   origPredictions,
                                                   protected_attribute,
                                                   protected_group,
                                                   header,
                                                   judgment)

            rerank.rerank_featurevectors(LSATRaceBlackData, self.p, post_process=True)

        if self.dataset == "engineering-NoSemi":

            """
            Engineering Students Data - NoSemi - gender

            """
            print("Start reranking of Engineering Students Data - No Semi Private - gender")
            protected_group = "female"
            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("post-processing for " + fold + " with " + self.p_classifier)
                origPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
                rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/FA-IR/" + self.p_classifier + "/predictions.pred"
                EngineeringData = DatasetDescription(rerankedPredictions,
                                                     origPredictions,
                                                     protected_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                rerank.rerank_featurevectors(EngineeringData, self.p, post_process=True)

                fold_count += 1

            """
            Engineering Students Data - NoSemi - highschool

            """
            print("Start reranking of Engineering Students Data - No Semi Private - highschool")
            protected_group = "highschool"
            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("post-processing for " + fold + " with " + self.p_classifier)
                origPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
                rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/FA-IR/" + self.p_classifier + "/predictions.pred"
                EngineeringData = DatasetDescription(rerankedPredictions,
                                                     origPredictions,
                                                     protected_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                rerank.rerank_featurevectors(EngineeringData, self.p, post_process=True)

                fold_count += 1

