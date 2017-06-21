import numpy as np
from imblearn.ensemble import BalanceCascade 
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalanceCascade 
from imblearn.ensemble import EasyEnsemble 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

dataset = pd.read_csv('Project Train Dataset.csv', delimiter=',')
dataset.to_csv('train_ds.csv', quoting=csv.QUOTE_NONE, sep=',', index=False, escapechar=' ')
cleaned_dataset = pd.read_csv('train_ds.csv', delimiter=',')

cleaned_dataset.columns = ['CUST_COD', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE',
       'BIRTH_DATE', 'PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP',
       'PAY_AUG', 'PAY_JUL', 'BILL_AMT_DEC', 'BILL_AMT_NOV',
       'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
       'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP',
       'PAY_AMT_AUG', 'PAY_AMT_JUL', 'DEFAULT_PAYMENT_JAN']

cleaned_dataset.describe()

# Deleting useless columns
cleaned_dataset.drop('SEX', axis=1, inplace=True)
cleaned_dataset.drop('EDUCATION', axis=1, inplace=True)
cleaned_dataset.drop('MARRIAGE', axis=1, inplace=True)
cleaned_dataset.drop('BIRTH_DATE', axis=1, inplace=True)
cleaned_dataset.drop('CUST_COD', axis=1, inplace=True)

# Training Data
X = cleaned_dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
y = cleaned_dataset.iloc[:, 19].values

# Select KBEST
selection = SelectKBest(k=18).fit(X,y)
X = selection.transform(X)
print(X.shape)
print(selection.scores_)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('Original dataset shape {}'.format(Counter(y)))

ee = EasyEnsemble()

X_res, y_res = ee.fit_sample(X_train, y_train)

print('Resampled dataset shape {}'.format(Counter(y_res[0])))

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

classifier = XGBClassifier(objective='binary:logistic')

parameters = {'learning_rate': [0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3], 'n_estimators' : [10, 20, 50, 100, 200, 1000], 'max_depth':[1,3,4,5,6,7,8,9,10], 'min_child_weight': [1,2,3,4,5,6,10], 'gamma': [0.0, 0.2, 0.1, 0.3], 'subsample': [0.5, 0.8, 1]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 10,
                           n_jobs = -1,
                           verbose=2
                           )
grid_search = grid_search.fit(X_res[0], y_res[0])
best_f1 = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_f1)
print(best_parameters)







