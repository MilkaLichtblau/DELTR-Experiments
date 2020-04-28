#!/bin/bash
# runs all trainings for LSAT data and saves result models into respective folders

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
PATH_TO_LSAT_DATASETS=$GIT_ROOT/data/LawStudents
RESULT_DIR=$GIT_ROOT/results/LawStudents 


echo ""

################################################################################

FOLD=gender

GAMMA_SMALL=200000 
GAMMA_LARGE=1000000

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train.txt $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train_RERANKED.txt $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train_RERANKED_PPlus.txt $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train_RERANKED_PMinus.txt $RESULT_DIR/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train.txt $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train.txt $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_train.txt $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

################################################################################

FOLD=race_black

GAMMA_SMALL=100000 
GAMMA_LARGE=500000

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $RESULT_DIR/$FOLD/COLORBLIND/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train.txt $RESULT_DIR/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
./train.m $RESULT_DIR/$FOLD/PREPROCESSED/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train_RERANKED.txt $RESULT_DIR/$FOLD/PREPROCESSED/model.m

./train.m $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train_RERANKED_PPlus.txt $RESULT_DIR/$FOLD/PREPROCESSED_PPlus/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR
./train.m $RESULT_DIR/$FOLD/GAMMA\=0/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train.txt $RESULT_DIR/$FOLD/GAMMA\=0/model.m 0

cp $RESULT_DIR/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$FOLD/FA-IR/model.m 

echo "$FOLD GAMMA=SMALL..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=SMALL/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train.txt $RESULT_DIR/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $RESULT_DIR/$FOLD/GAMMA\=LARGE/ $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_train.txt $RESULT_DIR/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

