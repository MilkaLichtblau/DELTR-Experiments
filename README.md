# DELTR-Experiments
Code and Datasets for our [paper](https://arxiv.org/abs/1805.08716) on disparate exposure in learning to rank. Use this repository for reproduction of the paper's experiments only. If you want to incorporate the method into your project, visit our [Fair Search](https://github.com/fair-search) repository, where the method is available as a library in [Java](https://github.com/fair-search/fairsearch-deltr-java), [Python](https://github.com/fair-search/fairsearch-deltr-python) and as [Elasticsearch interface](https://github.com/fair-search/fairsearch-deltr-for-elasticsearch).

## Requirements
requires GNU Octave and the packages ``octave-general`` and ``octave-parallel``

This code has been tested with Octave version 5.1.0 and Ubuntu 18.04. 

One known issue is a bug involving osmesa. The scripts already contain a workaround for this bug.

## Usage

### Data Preparation
The root directory contains two bash-scripts that prepare all datasets for the pre-, in-, and post-processing experiments: 

**1. prepareDatasetsForDELTR.sh:** prepares datasets for DELTR and post-processing experiments

**2. preprocess.sh:** prepares datasets for pre-processing experiments

You only need these, if you want to recreate the datasets, possibly because of altering parameters. The scripts are already run and all datasets are available in [data/](https://github.com/MilkaLichtblau/DELTR-Experiments/tree/master/data) 

### Model Training

For each dataset a bash-script named ``trainDATASET.sh`` is available that trains models for all experimental settings and saves them into ``results/DATASET/PROTECTED-ATTRIBUTE/FOLD/EXPERIMENTAL-SETTING/model.m``. 

To check sanity of the model, the code also creates a plot of the cost and gradient development, which should both convert. The figures are stored in the same folder as the model.

Training parameters like learning rate or number of iterations can be changed in [``listnet-src/globals.m``](https://github.com/MilkaLichtblau/DELTR-Experiments/blob/master/listnet-src/globals.m) and [``deltr-src/globals.m``](https://github.com/MilkaLichtblau/DELTR-Experiments/blob/master/deltr-src/globals.m). The only exception is the Gamma parameter for DELTR which is a command line argument. 

### Predictions

For each dataset a bash-script named ``predictDATASET.sh`` is available, that uses the previously trained models and testdata to predict rankings. Predictions are stored in the same folder as model.m.

This script also copies the prediction files from the DELTR experiment with Gamma=0 into folders ``DATASET/FA-IR/P-VALUE``, which are then needed by the post-processing script.

### Post-Processing with FA\*IR

Post-processes a potentially unfair ranking by applying the FA\*IR algorithm \([https://arxiv.org/pdf/1706.06368.pdf](https://arxiv.org/pdf/1706.06368.pdf)\) to each dataset. 

### Result Evaluation

All result plots and evaluation sheets are saved in ``results/DATASET/PROTECTED-ATTRIBUTE/results/`` (e.g. "results/LawStudents/gender/results"). Results are averaged across all queries, if applicable.
