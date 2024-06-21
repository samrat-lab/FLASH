# FLASH Algorithm

## Overview
FLASH (Feature Learning Augmented with Sampling and Heuristics) is a feature selection method designed for gene expression datasets. It aims to identify the most relevant features (genes) that contribute to distinguishing different classes in the dataset, thus improving the performance of machine learning models.

## Design
This repository contains Python scripts that guide you step by step through the process of feature selection, model fitting, and comparison. The results are saved according to the directory structure mentioned below, but users can customize the directory paths as needed by customizing the config file. The saved results from each step are used in subsequent steps to build upon the current results and move forward in the analysis. Please follow  [workflow notebook](workflow.ipynb) for running the scripts.

## Dataset
Please download the gene expression dataset from the Gene Expression Omnibus (GEO) and store it in a .csv format, ensuring that the last column contains the class labels. Additionally, six datasets are provided in the `data` folder, which are not available on GEO.

## Prerequisites
### Major Packages Required
* Pandas
* Numpy
* scikit-learn
* Matplotlib
  To install the packages use command ``` pip install package-name ```
### Directory structure
* code directory (example: exp1)
* Input/raw/exp1
* Input/prep/exp1
* Output/exp1/Results/


## Example Workflow (For one dataset)
* Clone the Repository:
  ```git clone https://github.com/samrat-lab/FLASH.git```
  Prepare config file for each dataset using [Sample file](config_ALL2.json) and set directory to desired location.
* Sampling and t-test to generate sampled t-test matrix
  Use the config file to obtain sampled t-test on a specific dataset [get_bs_ttest](get_bs_ttest.py).
* Calculating the score using sampled t-test matrix
  Use the matrix generated to calcuate the average value of significant feature and apply a threshold to filter important subset.
* Performing recurive feature elimination on filtered feature set
  Use function [perform_rfe](perform_rfe.py) to remove less important feature iteratively.
* Asses predictive performance of FLASH features
  Compare multiple ML models to obtain the preditive performance of FLASH features using [compare_ml_models.py](compare_ml_models)
* Repeat for other dataset
  Creating config file for other datasets and run the above steps in loop.

