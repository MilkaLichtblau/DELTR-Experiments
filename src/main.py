'''
Created on Apr 2, 2018

@author: meike.zehlike
'''
import argparse, os

from data_preparation import *
from evaluation.evaluate import DELTR_Evaluator
from processingWithFair.preprocess import Preprocessing
from processingWithFair.postprocess import Postprocessing


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='Disparate Exposure in Learning To Rank',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--create",
                        nargs=1,
                        metavar='DATASET',
                        choices=['law-gender',
                                 'law-black',
                                 'trec',
                                 'engineering-withoutSemiPrivate'],
                        help="creates datasets from raw data and writes them to disk")
    parser.add_argument("--evaluate",
                        nargs=1,
                        metavar='DATASET',
                        choices=['synthetic',
                                 'law-gender',
                                 'law-black',
                                 'trec',
                                 'engineering-gender-withoutSemiPrivate',
                                 'engineering-highschool-withoutSemiPrivate'],
                        help="evaluates performance and fairness metrics for DATASET predictions")
    parser.add_argument("--preprocess",
                        nargs=2,
                        metavar=("DATASET", "P"),
                        choices=['law',
                                 'trec',
                                 'engineering-NoSemi',
                                 'p_minus',
                                 'p_auto',
                                 'p_plus'],
                        help="reranks all folds for the specified dataset with FA*IR for pre-processing (alpha = 0.1)")
    parser.add_argument("--postprocess",
                        nargs=2,
                        metavar=("DATASET", "P"),
                        choices=['law',
                                 'trec',
                                 'engineering-NoSemi',
                                 'p_minus',
                                 'p_auto',
                                 'p_plus'],
                        help="reranks all folds for the specified dataset with FA*IR for post-processing (alpha = 0.1)")

    args = parser.parse_args()

    ################### argparse create #########################
    if args.create == ['law-gender']:
        train, test = lawStudPrep.prepareGenderData()
        train.to_csv('../data/LawStudents/gender/LawStudents_Gender_train.txt', index=False, header=False)
        test.to_csv('../data/LawStudents/gender/LawStudents_Gender_test.txt', index=False, header=False)
    elif args.create == ['law-black']:
        train, test = lawStudPrep.prepareRaceData('Black', 'White')
        train.to_csv('../data/LawStudents/race_black/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../data/LawStudents/race_black/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['trec']:
        TREC_prep.prepare()
    elif args.create == ['engineering-withoutSemiPrivate']:
        EngineeringStudPrep.prepareNoSemi()

    #################### argparse evaluate ################################
    elif args.evaluate == ['law-gender']:
        resultDir = '../results/LawStudents/gender/results/'
        binSize = 200
        protAttr = 1
        evaluator = DELTR_Evaluator('law-gender',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    elif args.evaluate == ['law-black']:
        resultDir = '../results/LawStudents/race_black/results/'
        binSize = 200
        protAttr = 1
        evaluator = DELTR_Evaluator('law-black',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    elif args.evaluate == ['trec']:
        resultDir = '../results/TREC/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('trec',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    elif args.evaluate == ['engineering-gender-withoutSemiPrivate']:
        resultDir = '../results/EngineeringStudents/NoSemiPrivate/gender/results/'
        binSize = 20
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-gender-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    elif args.evaluate == ['engineering-highschool-withoutSemiPrivate']:
        resultDir = '../results/EngineeringStudents/NoSemiPrivate/highschool/results/'
        binSize = 20
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-highschool-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    #################### argparse pre-process ################################
    elif (args.preprocess != None):
        if ('trec' in args.preprocess or 'law' in args.preprocess or 'engineering-NoSemi' in args.preprocess) and \
           ('p_minus' in args.preprocess or 'p_auto' in args.preprocess or 'p_plus' in args.preprocess) and \
           len(args.preprocess) == 2:
            preprocessor = Preprocessing(args.preprocess[0],
                                         args.preprocess[1])
            preprocessor.preprocess_dataset()

    #################### argparse post-process ################################
    elif (args.postprocess != None):
        if ('trec' in args.postprocess or 'law' in args.postprocess or 'engineering-NoSemi' in args.postprocess) and \
           ('p_minus' in args.postprocess or 'p_auto' in args.postprocess or 'p_plus' in args.postprocess) and \
           len(args.postprocess) == 2:
            postprocessor = Postprocessing(args.postprocess[0],
                                           args.postprocess[1])
            postprocessor.postprocess_result()
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
