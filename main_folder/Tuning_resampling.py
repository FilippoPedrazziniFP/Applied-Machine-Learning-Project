# Useful Libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# stat libraries
from scipy import stats

# Importing all classifiers from Scikitlearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import NuSVC

# stat libraries
from scipy import stats

# Libraries for the evaluation
from sklearn import model_selection
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Imbalanced Learning Library Imports
from collections import Counter
from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalanceCascade 
from imblearn.ensemble import EasyEnsemble 

# Another kind of Classifier
from xgboost import XGBClassifier

# Importing the dataset
dataset = pd.read_csv('Project Train Dataset.csv', delimiter=',')
dataset.to_csv('train_ds.csv', quoting=csv.QUOTE_NONE, sep=',', index=False, escapechar=' ')
cleaned_dataset = pd.read_csv('train_ds.csv', delimiter=',')

cleaned_dataset.columns = ['CUST_COD', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE',
       'BIRTH_DATE', 'PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP',
       'PAY_AUG', 'PAY_JUL', 'BILL_AMT_DEC', 'BILL_AMT_NOV',
       'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
       'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP',
       'PAY_AMT_AUG', 'PAY_AMT_JUL', 'DEFAULT_PAYMENT_JAN']



#print(cleaned_dataset.describe())


# Deleting useless columns
cleaned_dataset.drop('SEX', axis=1, inplace=True)
cleaned_dataset.drop('EDUCATION', axis=1, inplace=True)
cleaned_dataset.drop('MARRIAGE', axis=1, inplace=True)
cleaned_dataset.drop('BIRTH_DATE', axis=1, inplace=True)
cleaned_dataset.drop('CUST_COD', axis=1, inplace=True)

#cleaned_dataset['SUM_LATE'] = (cleaned_dataset.PAY_DEC + cleaned_dataset.PAY_NOV + cleaned_dataset.PAY_OCT + cleaned_dataset.PAY_SEP + cleaned_dataset.PAY_AUG + cleaned_dataset.PAY_JUL)
#cleaned_dataset['SUM_BILL'] = (cleaned_dataset.BILL_AMT_DEC + cleaned_dataset.BILL_AMT_NOV + cleaned_dataset.BILL_AMT_OCT + cleaned_dataset.BILL_AMT_SEP + cleaned_dataset.BILL_AMT_AUG + cleaned_dataset.BILL_AMT_JUL) 
#cleaned_dataset['SUM_PAY'] = (cleaned_dataset.PAY_AMT_DEC + cleaned_dataset.BILL_AMT_NOV + cleaned_dataset.BILL_AMT_OCT + cleaned_dataset.BILL_AMT_SEP + cleaned_dataset.BILL_AMT_AUG + cleaned_dataset.BILL_AMT_JUL) 



#cleaned_dataset = cleaned_dataset[['LIMIT_BAL','SUM_LATE', 'SUM_BILL', 'SUM_PAY', 'DEFAULT_PAYMENT_JAN']]
#cleaned_dataset.drop('PAY_DEC', axis=1, inplace=True)
#cleaned_dataset.drop('PAY_NOV', axis=1, inplace=True)
#cleaned_dataset.drop('PAY_OCT', axis=1, inplace=True)
#cleaned_dataset.drop('PAY_SEP', axis=1, inplace=True)
#cleaned_dataset.drop('PAY_AUG', axis=1, inplace=True)
#cleaned_dataset.drop('PAY_JUL', axis=1, inplace=True)

print(cleaned_dataset.head(5))

# Training Data
X = cleaned_dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
y = cleaned_dataset.iloc[:, 19].values

"""def featuresel(x,y,data):
    # Select KBEST
    selection = SelectKBest(k=13).fit(x,y)
    # a = selection.transform(x)
    features = data
    d = {'Feature': features , 'weight': selection.scores_}
    df = pd.DataFrame(data=d)
    df = df.sort_values(by='weight', ascending=0)
    print(df)
    return # a

target = 'DEFAULT_PAYMENT_JAN'
variables = cleaned_dataset.columns[cleaned_dataset.columns!=target]
X = cleaned_dataset[variables] 
y = cleaned_dataset[target]
featuresel(X,y,variables)"""

# Select KBEST
#selection = SelectKBest().fit(X,y)
#X = selection.transform(X)
#print(X.shape)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

xgb = XGBClassifier()
sgdc = SGDClassifier(alpha=0.00031000000000000005, class_weight='balanced', l1_ratio=0.70000000000000007, loss='perceptron', n_iter=4, penalty='l1')
dtc = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=3, min_samples_leaf=10, min_samples_split=2)
bnb = BernoulliNB()
xtc = ExtraTreesClassifier(class_weight='balanced', max_depth=12, min_samples_leaf=10, min_samples_split=2, n_estimators=220)
svc = SVC(probability=True)
nb = GaussianNB()
qda = QuadraticDiscriminantAnalysis()
lr = LogisticRegression()
lda = LinearDiscriminantAnalysis(shrinkage=0.70000000000000007, solver='eigen')
ada = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.5, n_estimators=28)
rf = RandomForestClassifier(class_weight='balanced', max_depth=12, min_samples_leaf=40, min_samples_split=4, n_estimators=220)
knn = KNeighborsClassifier()
gbc = GradientBoostingClassifier()
mlp = MLPClassifier(alpha=0.15, learning_rate_init=0.01)
#nusvc = NuSVC()

"""from sklearn.model_selection import GridSearchCV

c_param = np.arange(0.1,0.9, 0.1)
a_param = np.arange(0.1,0.9, 0.1)

parameters = [{'nu': c_param, 'kernel': ['linear']},
              {'nu': c_param, 'kernel': ['rbf'], 'gamma': a_param }]

grid_search = GridSearchCV(estimator = nusvc,
                           param_grid = parameters,
                           scoring='f1',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_f1 = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_f1)
print(best_parameters)"""



estimators = []
estimators.append(('gbc', gbc))
estimators.append(('svm', svc))
estimators.append(('rf', rf))
estimators.append(('mlp', mlp))
estimators.append(('xtc', xtc))
estimators.append(('xgb', xgb))
estimators.append(('ada', ada))
ens = VotingClassifier(estimators,voting='soft', n_jobs=-1)

models = []
#models.append(('LR', lr, '0.1'))
#models.append(('LDA', lda, '0.2'))
#models.append(('KNN', knn, 'Red'))
models.append(('CART', dtc, 'Red'))
#models.append(('NB', nb, '0.5'))
#models.append(('GBC', gbc, 'Blue'))
#models.append(('SVM', svc, 'Green'))
models.append(('RF', rf, 'Yellow')) 
models.append(('MLP', mlp, 'Cyan'))
models.append(('ADA', ada, 'Magenta'))
#models.append(('QDA', qda, '0.25'))
#models.append(('XGB', xgb, 'Black'))
#models.append(('SGDC', sgdc, 'Yellow'))
#models.append(('BNB', bnb, '0.55'))
models.append(('XTC', xtc, 'Blue'))
#models.append(('ENS', ens, '#808080'))
#models.append(('NUSVC', nusvc, ''))



ee = EasyEnsemble()
n_model = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
n_model = np.array(n_model)
# Transforming the lists into array for plotting
plt.figure()
plt.title('Feature Selection')
plt.xlabel("Number of K-Best Features")
plt.ylabel("F1 - Score")
plt.grid()

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(models)))

#fin_avg = []
# Select KBEST

for name, model, c in models:

	scores = []
		
	for n in range(2,19):

		selection = SelectKBest(k=n).fit(X,y)
		X_k = selection.transform(X_train)
		X_k_test = selection.transform(X_test)
		    
		# Reshape
		X_k_res, y_k_res = ee.fit_sample(X_k, y_train)

		model.fit(X_k_res[0], y_k_res[0])
		y_pred = model.predict(X_k_test)

		f1 = f1_score(y_test, y_pred)
		scores.append(f1)
		    
		print(f1)    
		
	scores = np.array(scores)
	avg = np.average(scores)
		
	print("MODEL: ", name, avg)  
	plt.plot(n_model, scores, 'o-', color=c, label=name)	    
#fin_avg = np.array(fin_avg)
#fin_avg = np.average(fin_avg)
#print("Final AVG: ", fin_avg)
plt.axis([0, 20 , 0.5, 0.57])
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
plt.show()


##### THRESHOLD

"""
threshold = np.arange(0.1,1,0.1)
print(threshold)
n_threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n_threshold = np.array(n_threshold)
scores = []

ee = EasyEnsemble()
# Transforming the lists into array for plotting
plt.figure()
plt.title('Feature Selection')
plt.xlabel("Number of K-Best Features")
plt.ylabel("F1 - Score")
plt.grid()

selection = SelectKBest(k=15).fit(X,y)
X_k = selection.transform(X_train)
X_k_test = selection.transform(X_test)
X_k_res, y_k_res = ee.fit_sample(X_k, y_train)
xgb.fit(X_k_res[0], y_k_res[0])
y_pred = xgb.predict_proba(X_k_test)


for th in threshold:
	
	y_pred_th = y_pred[:,1] > th
	f1 = f1_score(y_test, y_pred_th)
	scores.append(f1)

plt.plot(n_threshold, scores, 'o-', color='Red')
plt.axis([0, 1 , 0.5, 0.57])
#plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
plt.show()"""

"""
sme = SMOTEENN()
smt = SMOTETomek()
sm = SMOTE()
	# Ensemble samplers
bc = BalanceCascade()
ee = EasyEnsemble()

resampling = []
res_name = []
resampling.append(bc)
resampling.append(ee)
resampling.append(('bc', bc))
resampling.append(('ee', ee))

for resampl in resampling:

	print('Original dataset shape {}'.format(Counter(y)))
	# Create the Samplers and Using our OWN Classifier
	X_res, y_res = resampl.fit_sample(X_train, y_train)
	print('Resampled dataset shape {}'.format(Counter(y_res[0])))


	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
	    model.fit(X_res[0], y_res[0])
	    y_pred = model.predict(X_test)
	    result = f1_score(y_test, y_pred)
	    results.append(result)
	    names.append(name)
	    msg = "%s: %f " % (name, result)
	    print(msg)

	n = ['LR','LDA','KNN','CART','NB','GBC','SVM','RF','MLP','ADA','QDA','XGB','BNB', 'XTC']
	n_number = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	#n = np.array(n)
	results = np.array(results)
	plt.figure(figsize=(15, 8))
	plt.title('Algorithm Comparison')
	plt.xlabel("Classifiers")
	plt.ylabel("F1 - Score")
	plt.grid()
	plt.xticks(n_number, n)
	plt.plot(n_number, results, 'o-', color="red", label="Reshaped DS")
	plt.legend()
	plt.ylim([0.3,0.6])
	plt.show()"""