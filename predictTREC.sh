#!/bin/bash
# runs all predictions for W3C Experts data and saves results into respective folders

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/deltr-src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/listnet-src
PATH_TO_TREC_DATASETS=$GIT_ROOT/data/TREC
RESULT_DIR=$GIT_ROOT/results/TREC
TRAINING_SCORES_FILENAME=trainingScores_ORIG.pred
PREDICTIONS_FILENAME=predictions.pred
RERANKED_PREDICTIONS_FILENAME=predictions_ORIG.pred

echo ""
echo "################################# PREDICTING TREC #############################################"

FOLD=fold_1

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/


cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/

FOLD=fold_2

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/


cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/

FOLD=fold_3

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/

cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/



FOLD=fold_4

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/

cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/


FOLD=fold_5

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/

cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/


FOLD=fold_6

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m $RESULT_DIR/$FOLD/COLORBLIND/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m $RESULT_DIR/$FOLD/PREPROCESSED/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/

cd $PATH_TO_EXECUTABLE_DELTR
./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/GAMMA\=0/

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Star/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Plus/$RERANKED_PREDICTIONS_FILENAME

cp $RESULT_DIR/$FOLD/GAMMA\=0/$TRAINING_SCORES_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$TRAINING_SCORES_FILENAME
cp $RESULT_DIR/$FOLD/GAMMA\=0/$PREDICTIONS_FILENAME $RESULT_DIR/$FOLD/FA-IR/P-Minus/$RERANKED_PREDICTIONS_FILENAME

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/

./predict.m $PATH_TO_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/



