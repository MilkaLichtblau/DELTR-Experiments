#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# script to prepare all datasets for DELTR experiments from paper 

cd src/

# The following line only works with the preparation code and the original dataset available, to which access is restricted due to privacy concerns.
# If you want to work with the original engineering students dataset, please contact the authors.
# python3 main.py --create engineering-withoutSemiPrivate 

python3 main.py --create trec
python3 main.py --create law-gender
python3 main.py --create law-black

cd ../
