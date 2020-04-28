#!/bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd src/

# rerank Chile SAT dataset folds using FA*IR for different values of p
python3 main.py --postprocess engineering-NoSemi p_auto
python3 main.py --postprocess engineering-NoSemi p_plus
python3 main.py --postprocess engineering-NoSemi p_minus

# rerank LSAT dataset folds using FA*IR for different values of p
python3 main.py --postprocess law p_auto
python3 main.py --postprocess law p_plus
python3 main.py --postprocess law p_minus

# rerank trec folds using FA*IR for different values of p
python3 main.py --postprocess trec p_auto
python3 main.py --postprocess trec p_plus
python3 main.py --postprocess trec p_minus

cd ../
