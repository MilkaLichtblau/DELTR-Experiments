#!/bin/bash
# runs all trainings for big TREC data and saves result models into respective folders

# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

export LD_PRELOAD=libGLX_mesa.so.0 	#very dirty hack to workaround this octave bug: error: __osmesa_print__: 
					#Depth and stencil doesn't match, are you sure you are using OSMesa >= 9.0?

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/deltr-src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/listnet-src
PATH_TO_TREC_DATASET=$GIT_ROOT/data/TREC 
RESULT_DIR=$GIT_ROOT/results/TREC 

GAMMA_SMALL=20000
GAMMA_LARGE=40000

echo ""
###############################################################################

FOLD=fold_1

#echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."

cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

################################################################################

FOLD=fold_2

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

##################################################################################

FOLD=fold_3

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

###################################################################################

FOLD=fold_4

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

####################################################################################

FOLD=fold_5

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $PATH_TO_TRERESULT_DIRC_DATASET/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

####################################################################################

FOLD=fold_6

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_TREC_DATASET/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE




