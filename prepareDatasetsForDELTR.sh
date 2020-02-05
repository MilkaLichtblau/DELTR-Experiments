#!/bin/bash
# script to prepare all datasets for DELTR experiments from paper 

cd src/

# The following line only works with the preparation code and the original dataset available, to which access is restricted due to privacy concerns.
# If you want to work with the original engineering students dataset, please contact the authors.
# python3 main.py --create engineering-withoutSemiPrivate 

python3 main.py --create trec
python3 main.py --create law-gender
python3 main.py --create law-black

cd ../
