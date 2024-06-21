#!/usr/bin/env python
# coding: utf-8

# In[1]:


# the libraries we need

#Models

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


#Metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn import preprocessing




#Utility
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from glob import glob
from os import makedirs as mkd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import pandas as pd
import numpy as np

#import imblearn
import time
import pickle
import argparse
import json
import warnings




warnings.filterwarnings("ignore", message=".*the default evaluation metric used with the objective.*")




def intersection(lst1, lst2):
    
    '''
    Intersection function returns common values in two list (lst1 & lst2)
    '''
    return list(set(lst1) & set(lst2))

def fetch_data(exp_fname='GSE162694_norm.csv',gene_fname='FDR_list_cutoff_old.05.xlsx',cat_col="labels"):
    
    '''
    fetch_data generates a dataframe for specific features or column and creates a label column at the end of the dataframe
    '''

    data=pd.read_csv(exp_fname)
    high_gene=pd.read_csv(gene_fname)
 
    ready_data=data[high_gene['Genes'].to_list()+[cat_col]]
    return ready_data



def get_models(rnd=24):
    
    
    '''
    Initializes the model required for evaulation
    '''
    
    models = dict()
    
    # decision tree
    model = DecisionTreeClassifier(random_state=rnd)
    models['dt'] = model
    
    # random forest
    model = RandomForestClassifier(random_state=rnd,n_estimators=50)
    models['rf'] = model

    #KNN
    model = KNeighborsClassifier()
    models['knn'] = model
    
    
    # SVM
    model = SVC(kernel='linear',random_state=rnd)
    models['svm'] = model
    
    # XGBoost
    model = XGBClassifier(use_label_encoder=False, random_state=rnd)
    models['xgb'] = model
    
    # Logistic regression
    model = LogisticRegression(solver='lbfgs',max_iter=1000,random_state=rnd)
    models['logit'] = model
    return models


# In[6]:





def get_viz(inp_df,title,grp='test'):
    
    '''
    Generates box plot for all the compared ML algorithm using a dataframe
    '''
    
    labels={'dt':'Decision Tree','rf':'Random Forest','knn':'KNN','svm':'SVM','xgb':'XGBoost','logit':'Logistic Regression'}
    inp_df.replace({"Algo":labels},inplace=True)
    viz_df=inp_df.filter(like=grp)
    viz_df.loc[:,'Algo']=inp_df['Algo']
    fig = px.box(
        viz_df,color='Algo',title=title,
        labels={}
                )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title='Scores',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0,1, 2, 3, 4],
            ticktext = ['Accuracy','F1-Score','AUROC','Precision','Recall']
        ),
        font = dict(
            size=24,
            color='black'
            #family='Arial Black'
        ),
        #template='simple_white',
        width=1000,
        height=600,
        legend=dict(
        orientation="h",    
        title='Legends'
        )
    )
    return fig


def get_res_summary(FDR_res,f_name,fig_title='High Gene List'):
    
    '''
    Calculates the average result out of cross-valdated performance metrics for each model
    '''
    

    output_file=f'{out_src}model_comparision{f_name}_avg.xlsx'

    temp_ls=[]
    for name,my_entry in FDR_res.items():# Iterating for each algorithm
        df=pd.DataFrame(my_entry)
        df['Algo']=name
        temp_ls.append(df)
    my_df = pd.concat(temp_ls)
    
    
    #saving the average result in xlsx file
    my_df.drop(['fit_time','score_time','estimator'],axis=1,inplace=True)
    with pd.ExcelWriter(output_file) as writer:

        df_mean=my_df.groupby('Algo').mean()
        df_std=my_df.groupby('Algo').std()
        df_mean.to_excel(writer,sheet_name='mean')
        df_std.to_excel(writer,sheet_name='std')

    return_fig2  = get_viz(my_df,f'{fig_title} Test dataset',grp='test')
    return_fig1  = get_viz(my_df,f'{fig_title} Train dataset',grp='train')
    

    return return_fig1, return_fig2

    

##########--Start--Preparation required variables ##############
parser = argparse.ArgumentParser()
parser.add_argument('--inp_file',type=str)
parser.add_argument('--fig_file',type=bool,default=False)
parser.add_argument('--gene_file',type=str)
parser.add_argument('--res_file',type=str)
args = parser.parse_args()

with open(args.inp_file) as f:
    configs = json.load(f)

cat1 = configs['cat1']
cat2 = configs['cat2']
cat_col = configs['cat_col']
inp_src = configs['inp_src']
out_src = configs['out_src']
exp_fname = configs['exp_fname']
rnd = configs['rnd_seed']
f_title = configs['fig_title']
gene_fname = args.gene_file
f_name = args.res_file

le = preprocessing.LabelEncoder()
print('This is encoding--',cat1,'--',cat2)
le.fit([cat1, cat2])


mkd(out_src,exist_ok=True)
output_file=f'{out_src}model_comparision{f_name}.xlsx'
##########--End--Preparation required variables ##############
    

##########--Start--Data loading and preparation##############


dataset = fetch_data(exp_fname,gene_fname,cat_col)

X=dataset.iloc[:,0:-1]
print("Dimension of input data ",X.shape)
labels=dataset.iloc[:,-1]
y=le.transform(labels)

##########--End--Data loading and preparation##############




##########--Start--Data loading and preparation##############
models = get_models(rnd)

results={}
with pd.ExcelWriter(output_file) as writer:


    for name, model in tqdm(models.items()):

        rkf=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=rnd)
        cross_out = cross_validate(model, X, y,return_train_score=True,return_estimator=True ,scoring=('accuracy','f1','roc_auc','precision','recall'),n_jobs=24 ,cv=rkf)
        results[name]=cross_out
        result_df=pd.DataFrame(cross_out)
        result_df.to_excel(writer, sheet_name=name)
        
pickle.dump( results, open( f'{out_src}{f_name}.p', "wb" ) )

##########--Start--Data loading and preparation##############


# In[24]:


my_fig1, my_fig2 = get_res_summary(results,f_name,f_title)
if args.fig_file:
    pio.write_image(my_fig1,f'{out_src}train_model_comparison_{f_name}.png',scale=10)
    pio.write_image(my_fig2,f'{out_src}test_model_comparison_{f_name}.png',scale=10)
#my_fig.show()



# In[10]:

