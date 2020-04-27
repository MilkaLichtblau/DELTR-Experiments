#!/bin/bash

cd src/

python3 main.py --evaluate trec
#python3 main.py --evaluate engineering-highschool-withoutSemiPrivate
#python3 main.py --evaluate engineering-gender-withoutSemiPrivate
#python3 main.py --evaluate law-gender
#gender main.py --evaluate law-black

cd ../
