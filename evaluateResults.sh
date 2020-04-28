#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
cd src/

python3 main.py --evaluate trec
python3 main.py --evaluate engineering-highschool-withoutSemiPrivate
python3 main.py --evaluate engineering-gender-withoutSemiPrivate
python3 main.py --evaluate law-gender
python3 main.py --evaluate law-black

cd ../
