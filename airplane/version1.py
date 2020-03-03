# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:02:44 2020

@author: ashish
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier as GDB

df = pd.read_csv("G:/DataVisualisation/airplane/train.csv", encoding = 'utf-8')
dfTest = pd.read_csv("G:/DataVisualisation/airplane/test.csv")

#Assesing feature importance using random forest 

forest = RandomForestClassifier(n_estimators=500,random_state=1)  

X = df.iloc[:,1:].values
y = df.iloc[:,0].values
X_test = dfTest.values


classLbl = LabelEncoder()
std = StandardScaler()
y = classLbl.fit_transform(y)

forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

feat_labels = df.columns[1:]

for f in range(X.shape[1]):
    print("%2d) %-*s %f"%(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]),feat_labels[indices],rotation = 90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

def modify_df(df):
    # Selecting The most important features
    df_mod = df.loc[:,feat_labels[indices[:5]]]
    
    #Adding dummy columns to categorical features
    
    df_mod = pd.concat([df_mod.drop('Accident_Type_Code', axis = 1), 
                       pd.get_dummies(df_mod['Accident_Type_Code'])], axis = 1)
        
    col_app = ["Acc_Type"+str(i) for i in range(1,8)]
    ls = df_mod.columns[:4]
    ls = list(ls)+col_app
    df_mod.columns = ls
    X_mod = df_mod.values
    return X_mod

def modify_df_keep_all(df):
    
    
    df_mod = pd.concat([df.drop('Accident_Type_Code', axis = 1), 
                       pd.get_dummies(df['Accident_Type_Code'], prefix='Acc_type')], axis = 1)
    
    feat_label = list(df_mod.columns)
    if "Severity" in feat_label:
        feat_label.remove("Severity")
        
    X_mod = df_mod.loc[:,feat_label].values
    X_mod[:,0] = std.fit_transform(X_mod[:,0].reshape(-1,1)).reshape(X_mod.shape[0])
    return X_mod



def fit_xgb_rand_search(X, Y):
    params = {
        'min_child_weight': [1,2,3,4,5,6,7,8,9,10],
        'gamma': [0.5,0.6,0.7,0.8,0.9, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.75, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate':np.linspace(0.01,0.1,5),
        'n_estimators':[500]
        }
    xgb = XGBClassifier()
    folds = 10
    param_comb = 10
    
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    fone_scorer = make_scorer(fbeta_score, beta=1, average = 'weighted')
    random_search = RandomizedSearchCV(xgb, param_distributions = params, n_iter=param_comb,
                                       scoring=fone_scorer, n_jobs=4, 
                                       cv=skf.split(X,Y), verbose=3, 
                                       random_state=1001 )
    random_search.fit(X, Y)
    return random_search

def fit_ada_grid_search(X, Y):
    params = {
        'n_estimators' : [50,100,200,400,500],
        'learning_rate' :np.linspace(0.01,0.1,5),
        'algorithm' : ['SAMME']
        }
    ada =  AdaBoostClassifier()
    folds = 10
    
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    fone_scorer = make_scorer(fbeta_score, beta=1, average = 'weighted')
    grid_search = GridSearchCV(ada, param_grid = params,
                               scoring=fone_scorer, n_jobs=5, 
                               cv=skf.split(X,Y), verbose=3)
    grid_search.fit(X, Y)
    return grid_search

def fit_bag_grid_search(X,Y):
    params = {
        'base_estimator' : [None, SVC()],
        'n_estimators' : [10,50,100,150,200],
        'oob_score':[True,False]
        }
    
    bag = BaggingClassifier()
    folds = 10
    
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    fone_scorer = make_scorer(fbeta_score, beta=1, average = 'weighted')
    grid_search = GridSearchCV(bag, param_grid = params,
                               scoring=fone_scorer, n_jobs=5, 
                               cv=skf.split(X,Y), verbose=3)
    
    grid_search.fit(X,Y)
    learning_curve(grid_search.best_estimator_,X,Y)
    return grid_search
    
def fit_gdb_grid_search(X,Y):
    parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3,5,8],
    "n_estimators":[100,200,300,500]
    }
    
    bag = GDB()
    folds = 10
    
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    fone_scorer = make_scorer(fbeta_score, beta=1, average = 'weighted')
    grid_search = GridSearchCV(bag, param_grid = params,
                               scoring=fone_scorer, n_jobs=5, 
                               cv=skf.split(X,Y), verbose=3)
    
    grid_search.fit(X,Y)
    learning_curve(grid_search.best_estimator_,X,Y)
    return grid_search
      
    
X_mod = modify_df(df)
search = fit_gdb_grid_search(X_mod, y)
model = search.best_estimator_
model.fit(X_mod,y)

X_test_mod = modify_df(dfTest)
y_pred = model.predict(X_test_mod)

def make_res_df(y,dfTest):
    y = classLbl.inverse_transform(y)
    df_res = pd.DataFrame()
    df_res['Accident_ID'] = dfTest['Accident_ID']
    df_res['Severity'] = y
    df_res[['Accident_ID', 'Severity']].to_csv('predictions.csv',
          encoding='utf-8', index = False)
    
make_res_df(y_pred,dfTest)  


