#Models
from sklearn.feature_selection import SelectFpr, SelectFdr, SelectFromModel, SelectKBest, SelectPercentile, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

#Metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.utils import resample
from time import time

#Utility
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, variation
from scipy.stats import ttest_1samp
from scipy.stats import chisquare as chi
from joblib import Parallel, delayed
from os import listdir as ls 
from os import makedirs as mkd
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import json
import warnings
import pickle
import argparse



# Parsing the command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--inp_file',type=str)
args = parser.parse_args()

#Loading the config file
with open(args.inp_file) as f:
    configs = json.load(f)

cat1 = configs['cat1']
cat2 = configs['cat2']
cat_col = configs['cat_col']
inp_src = configs['inp_src']
out_src = configs['out_src']
f_name = configs['res_file']
exp_fname = configs['exp_fname']
gene_fname = configs['gene_fname']
rnd = configs['rnd_seed']
f_title = configs['fig_title']
score_file = args.inp_file.split('_')[-1].split('.')[0]


# Reading the expression dataset
dataset=pd.read_csv(exp_fname,index_col=0)


def get_pval(df_x,df_y,label1=0,label2=1): 
    '''
    Calculating p-value for the dataset using student's t-test
    
    input arguments
    df_x: data_matrix, where row and column corresponds to sample and genes/features 
    df_y: corresponding phenotype label/class for the samples in df_x
    
    output arguments
    t_statistic: inferential statistics
    p_value : a column vector containing p_value for each gene/feature
    
    '''
    
    t_statistic, p_value = ttest_ind(df_x.iloc[df_y.iloc[:,0].values==label1], df_x.iloc[df_y.iloc[:,0].values==label2])
    t_scores = np.abs(t_statistic)
    
    return t_statistic, p_value


def bs_ttest(i,label1 = "Healthy",label2 = "T1D"):
    '''
    Generating a random subset with repetition from the original data
    
    output argument
    return_pval: a matrix with rows and columns corresponding to the sampling number and total number of features.
    '''
    
    
    df_x=dataset.iloc[:,0:-1]
    df_y=dataset.iloc[:,-1]
    
    df_x_temp,df_y_temp = resample(df_x,df_y, n_samples=round(len(df_y)*0.66), replace=False, stratify=df_y)
    df_x_final,df_y_final = resample(df_x_temp,df_y_temp, n_samples=len(df_y), replace=True, stratify=df_y_temp)
    
    
    t_statistic, p_value = ttest_ind(df_x_final.iloc[df_y_final.values==label1], df_x_final.iloc[df_y_final.values==label2],random_state=1)
    
    return_pval =pd.Series(-np.log10(p_value).astype(np.float16),index=df_x.columns.to_list())
    
    #eturn  -np.log10(p_value)
    return return_pval
    


# Runing main code block to generate the bootstrap test matrix with p-values
st=time()
np.random.seed(rnd)
par_out1 = Parallel(n_jobs=20)(delayed(bs_ttest)(i,cat1,cat2) for i in range(1000))
print('Total time taken',time()-st)


# In[9]:

# Converting the generated matrix to dataframe
pval_df = pd.DataFrame(par_out1)


# Saving the generated dataframe # Remove inp_src to save file in working directory, or replace it with desired location
pval_df.to_csv(inp_src+f'bs_test_{score_file}.csv',index=False)

