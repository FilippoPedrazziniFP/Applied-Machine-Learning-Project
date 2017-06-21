import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split as tts
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection as ms
from sklearn import datasets, metrics, tree
from imblearn import over_sampling as os
from imblearn import pipeline as pl


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalanceCascade 
from imblearn.ensemble import EasyEnsemble 

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

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
"""selection = SelectKBest().fit(X,y)
X = selection.transform(X)
print(X.shape)
print(selection.scores_)"""

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Applying PCA
"""from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)"""


###### UNDER/OVER SMAPLING AND USE THE SCIKITLEARN ENSEMBLE

print('Original dataset shape {}'.format(Counter(y)))

# Create the Samplers and Using our OWN Classifier
sme = SMOTEENN()
smt = SMOTETomek()
sm = SMOTE()
bc = BalanceCascade()
ee = EasyEnsemble()


#enn = EditedNearestNeighbours()
#renn = RepeatedEditedNearestNeighbours()

X_res, y_res = ee.fit_sample(X_train, y_train)

#reshaped_ds = pd.concat([X_res, y_res], axis=1, ignore_index=True)
#reshaped_ds.to_excel('train_reshaped.xlsx')

print('Resampled dataset shape {}'.format(Counter(y_res[0])))



# Fitting the different classifiers on the Sampled DS
xgb = XGBClassifier(learning_rate =0.1,
                   n_estimators=20,
                   max_depth=10,
                   min_child_weight=2,
                   subsample=0.9,
                   gamma=0.3,
                   colsample_bytree=0.9,
                   objective= 'binary:logistic',
                   nthread=-1,
                   scale_pos_weight=1,
                   seed=27)
xgb.fit(X_res[0], y_res[0])

sgdc = SGDClassifier(alpha=0.00041, class_weight='balanced', l1_ratio=0.45, loss='hinge', n_iter=9, penalty='l1')
sgdc.fit(X_res[0], y_res[0])

dtc = DecisionTreeClassifier(class_weight='balanced', max_depth=5, max_features=9, min_samples_leaf=10, min_samples_split=6)
dtc.fit(X_res[0], y_res[0])

bnb = BernoulliNB(alpha=0.0, binarize=0.15)
bnb.fit(X_res[0], y_res[0])

xtc = ExtraTreesClassifier(class_weight='balanced', max_depth=7, max_features=7, n_estimators=360)
xtc.fit(X_res[0], y_res[0])

svc = SVC(C=1.0, gamma='auto', kernel = 'rbf', probability=True)
svc.fit(X_res[0], y_res[0])

nb = GaussianNB()
nb.fit(X_res[0], y_res[0])

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_res[0], y_res[0])

lr = LogisticRegression(C=0.001, penalty='l2')
lr.fit(X_res[0], y_res[0])

lda = LinearDiscriminantAnalysis(shrinkage=0.7, solver='eigen')
lda.fit(X_res[0], y_res[0])

ada = AdaBoostClassifier()
ada.fit(X_res[0], y_res[0])

rf = RandomForestClassifier(class_weight='balanced_subsample',max_features=5, n_estimators=900, max_depth=9)
rf.fit(X_res[0], y_res[0])

knn = KNeighborsClassifier(leaf_size=20, n_neighbors=150, weights='distance')
knn.fit(X_res[0], y_res[0])

gbc = GradientBoostingClassifier(min_samples_split=5, min_samples_leaf=3, n_estimators=70)
gbc.fit(X_res[0], y_res[0])

mlp = MLPClassifier(alpha=0.08, learning_rate_init=0.01)
mlp.fit(X_res[0], y_res[0])


# Add one transformers and two samplers in the pipeline object
#pipeline = make_pipeline(sm, classifier)
#pipeline.fit(X_train, y_train)

# Predicting with the diffent classifiers
y_pred_sgdc = svc.predict(X_test)
print("SGDC")
print(classification_report(y_test, y_pred_sgdc))

y_pred_bnb = svc.predict(X_test)
print("BNB")
print(classification_report(y_test, y_pred_bnb))

y_pred_dtc = svc.predict(X_test)
print("DTC")
print(classification_report(y_test, y_pred_dtc))

y_pred_xgb = svc.predict(X_test)
print("XGB")
print(classification_report(y_test, y_pred_xgb))

y_pred_xtc = svc.predict(X_test)
print("XTC")
print(classification_report(y_test, y_pred_xtc))

y_pred_svc = svc.predict(X_test)
print("SVC")
print(classification_report(y_test, y_pred_svc))


y_pred_nb = nb.predict(X_test)
print("NB")
print(classification_report(y_test, y_pred_nb))


y_pred_qda = qda.predict(X_test)
print("QDA")
print(classification_report(y_test, y_pred_qda))


y_pred_lr = lr.predict(X_test)
print("LR")
print(classification_report(y_test, y_pred_lr))


y_pred_lda = lda.predict(X_test)
print("LDA")
print(classification_report(y_test, y_pred_lda))


y_pred_ada = ada.predict(X_test)
print("ADA")
print(classification_report(y_test, y_pred_ada))

y_pred_rf = rf.predict(X_test)
print("RF")
print(classification_report(y_test, y_pred_rf))

y_pred_knn = knn.predict(X_test)
print("KNN")
print(classification_report(y_test, y_pred_knn))

y_pred_gbc = ada.predict(X_test)
print("GBC")
print(classification_report(y_test, y_pred_gbc))

y_pred_mlp = rf.predict(X_test)
print("MLP")
print(classification_report(y_test, y_pred_mlp))


# Comparing two vectors
from sklearn.metrics.pairwise import euclidean_distances
distance = euclidean_distances(y_pred_mlp, y_pred_rf)

print(y_pred_mlp)
print(y_pred_svc)
print(y_pred_rf)

print(distance)


# Ensemble classifier
# Voting Ensemble for Classification



###### ENSEMBLE CLASSIFIER - Majority Voting


seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# create the sub models
estimators = []
estimators.append(('rf', rf))
estimators.append(('mlp', mlp))
estimators.append(('svc', svc))


# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_res[0], y_res[0])

y_pred_en = ensemble.predict(X_test)
print("ENSEMBLE")
print(classification_report(y_test, y_pred_en))







######## PRECISION RECALL CURVE

"""
# Libraries for the evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

threshold = 0.5

# SVC
yprob_svc = svc.predict_proba(X_test)
precision_svc, recall_svc, thresholds_svc = precision_recall_curve(y_true=y_test, probas_pred=yprob_svc[:,1])

yprob_svc_th = yprob_svc[:,1] > threshold
f1_svc = f1_score(y_test, yprob_svc_th)
print("SVC")
print("F1", f1_svc)

# GBC
yprob_gbc = gbc.predict_proba(X_test)
precision_gbc, recall_gbc, thresholds_gbc = precision_recall_curve(y_true=y_test, probas_pred=yprob_gbc[:,1])

yprob_gbc_th = yprob_gbc[:,1] > threshold
f1_gbc = f1_score(y_test, yprob_gbc_th)
print("GBC")
print("F1", f1_gbc)


# XGB
yprob_xgb = xgb.predict_proba(X_test)
precision_xgb, recall_xgb, thresholds_xgb = precision_recall_curve(y_true=y_test, probas_pred=yprob_xgb[:,1])

yprob_xgb_th = yprob_xgb[:,1] > threshold
f1_xgb = f1_score(y_test, yprob_xgb_th)
print("XGB")
print("F1", f1_xgb)

# LDA
yprob_lda = lda.predict_proba(X_test)
precision_lda, recall_lda, thresholds_lda = precision_recall_curve(y_true=y_test, probas_pred=yprob_lda[:,1])

yprob_lda_th = yprob_lda[:,1] > threshold
f1_lda = f1_score(y_test, yprob_lda_th)
print("LDA")
print("F1", f1_lda)

# KNN
yprob_knn = knn.predict_proba(X_test)
precision_knn, recall_knn, thresholds_knn = precision_recall_curve(y_true=y_test, probas_pred=yprob_knn[:,1])

yprob_knn_th = yprob_knn[:,1] > threshold
f1_knn = f1_score(y_test, yprob_knn_th)
print("KNN")
print("F1", f1_knn)

# RF
yprob_rf = rf.predict_proba(X_test)
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_true=y_test, probas_pred=yprob_rf[:,1])

yprob_rf_th = yprob_rf[:,1] > threshold
f1_rf = f1_score(y_test, yprob_rf_th)
print("RF")
print("F1", f1_rf)


# LR
yprob_lr = lr.predict_proba(X_test)
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_true=y_test, probas_pred=yprob_lr[:,1])

yprob_lr_th = yprob_lr[:,1] > threshold
f1_lr = f1_score(y_test, yprob_lr_th)
print("LR")
print("F1", f1_lr)

yprob_dt = dt.predict_proba(X_test)
precision_dt, recall_dt, thresholds_dt = precision_recall_curve(y_true=y_test, probas_pred=yprob_dt[:,1])

yprob_dt_th = yprob_dt[:,1] > threshold
f1_dt = f1_score(y_test, yprob_dt_th)
print("DT")
print("F1", f1_dt)

# NB
yprob_nb = nb.predict_proba(X_test)
precision_nb, recall_nb, thresholds_nb = precision_recall_curve(y_true=y_test, probas_pred=yprob_nb[:,1])

yprob_nb_th = yprob_nb[:,1] > threshold
f1_nb = f1_score(y_test, yprob_nb_th)
print("NB")
print("F1", f1_nb)

# ADA
yprob_ada = ada.predict_proba(X_test)
precision_ada, recall_ada, thresholds_ada = precision_recall_curve(y_true=y_test, probas_pred=yprob_ada[:,1])

yprob_ada_th = yprob_ada[:,1] > threshold
f1_ada = f1_score(y_test, yprob_ada_th)
print("ADA")
print("F1", f1_ada)

# QDA
yprob_qda = qda.predict_proba(X_test)
precision_qda, recall_qda, thresholds_qda = precision_recall_curve(y_true=y_test, probas_pred=yprob_qda[:,1])

yprob_qda_th = yprob_qda[:,1] > threshold
f1_qda = f1_score(y_test, yprob_qda_th)
print("QDA")
print("F1", f1_qda)

plt.figure(1, figsize=(8, 6));
font = {'family':'sans', 'size':24};
plt.rc('font', **font);
plt.plot(recall_lr, precision_lr, label="Logistic Regression");
plt.plot(recall_svc, precision_svc, label="SVC");
plt.plot(recall_xgb, precision_xgb, label="XGBoost");
plt.plot(recall_lda, precision_lda, label="LDA");
plt.plot(recall_knn, precision_knn, label="KNN");
plt.plot(recall_gbc, precision_gbc, label="GBC");
#plt.plot(recall_dt, precision_dt, label="Decision Trees");
plt.plot(recall_nb, precision_nb, label="Naive Bayes");
plt.plot(recall_ada, precision_ada, label="ADABoost");
plt.plot(recall_qda, precision_qda, label="QDA");
plt.xlabel('Recall');
plt.ylabel('Precision');
plt.ylim([0.5,1.1])
plt.yticks(np.arange(0,1,.1))
plt.title('Precision-Recall Curve');
plt.plot(recall_lr[:-1],thresholds_lr, label="Threshold");
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
"""

######## NUMBER OF FEATURES

from sklearn.metrics import f1_score

###### USING STACKING CLASSIFIER
from mlxtend.classifier import StackingClassifier

xgb = XGBClassifier(learning_rate =0.01,
                       n_estimators=20,
                       max_depth=6,
                       min_child_weight=5,
                       gamma=0,
                       subsample=0.8,
                       base_score=0.5,
                       colsample_bytree=0.8,
                       objective= 'binary:logistic',
                       nthread=1,
                       scale_pos_weight=1,
                       seed=27)
svc = SVC(C=1.0, gamma='auto', kernel = 'rbf', probability=True)
qda = QuadraticDiscriminantAnalysis()
lr = LogisticRegression(C=0.001, penalty='l2')
ada = AdaBoostClassifier()
rf = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(150)
gbc = GradientBoostingClassifier()
mlp = MLPClassifier(alpha=1)
xtc = ExtraTreesClassifier(class_weight='balanced', max_depth=7, max_features=7, n_estimators=360)

sclf = StackingClassifier(classifiers=[rf, mlp, svc], 
                          meta_classifier=lr)

estimators = []
estimators.append(('rf', rf))
estimators.append(('mlp', mlp))
estimators.append(('svc', svc))

ensemble = VotingClassifier(estimators)


scores_mv = []
scores_st = []
# Select KBEST
for n in range(2,20):
    selection = SelectKBest(k=n).fit(X,y)
    X_k = selection.transform(X_train)
    X_k_test = selection.transform(X_test)
    X_k_res, y_k_res = bc.fit_sample(X_k, y_train)


    sclf.fit(X_k_res[0], y_k_res[0])
    y_pred_st = sclf.predict(X_k_test)

    f1_st = f1_score(y_test, y_pred_st)
    scores_st.append(f1_st)

    print(f1_st)

    ensemble.fit(X_k_res[0], y_k_res[0])
    y_pred_en = ensemble.predict(X_k_test)

    f1_en = f1_score(y_test, y_pred_en)
    scores_mv.append(f1_en)

    print(f1_en)
    

n = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# Transforming the lists into array for plotting
n = np.array(n)
scores_mv = np.array(scores_mv)
scores_st = np.array(scores_st)

plt.figure()
plt.title('Feature Selection')
plt.xlabel("Number of K-Best Features")
plt.ylabel("F1 - Score")
plt.grid()
plt.plot(n, scores_mv, 'o-', color="blue",
             label="Ensemble")
plt.plot(n, scores_st, 'o-', color="red",
             label="Stacking")
plt.axis([0, 20, 0.5, 0.6])
plt.show()




"""###### MAKING OUR ENSEMBLE CLASSIFIER

estimators = []
model1 = RandomForestClassifier()
estimators.append(('rf', model1))
model2 = AdaBoostClassifier()
estimators.append(('ada', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = KNeighborsClassifier(150)
estimators.append(('knn', model4))
model5 = GradientBoostingClassifier()
estimators.append(('gbc', model5))
model6 = XGBClassifier()
estimators.append(('gbc', model6))

ensemble = VotingClassifier(estimators)


# sm = BalanceCascade(random_state=42, estimator='linear-svm')
# building the majority vote classifier
print('Original dataset shape {}'.format(Counter(y_train)))

bc = BalanceCascade(random_state=RANDOM_STATE, estimator='linear-svm', n_max_subset=5)
ee = EasyEnsemble(random_state=RANDOM_STATE)


X_res_en, y_res_en = bc.fit_sample(X_train, y_train)

print('Resampled dataset shape {}'.format(Counter(y_res_en[0])))
print(X_res_en.shape)"""



# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and F1 score
"""from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

binary = f1_score(y_test, y_pred)  
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ")
print(cm)
print("F1 Measure: ", binary)"""
















####### VALIDATION CURVE









# Validation Curve with SMOTE
"""scorer = metrics.make_scorer(metrics.f1_score)

smote = os.SMOTE(random_state=RANDOM_STATE)
pipeline = pl.make_pipeline(smote, ensemble)

param_range = range(1, 11)
train_scores, test_scores = ms.validation_curve(
    pipeline, X_train, y_train, param_name="smote__k_neighbors", param_range=param_range,
    cv=3, scoring=scorer, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(param_range, test_scores_mean, label='SMOTE')
ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)
idx_max = np.argmax(test_scores_mean)
plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
            label=r'F1 Score: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))

plt.title("Validation Curve with SMOTE-ENSEMBLE")
plt.xlabel("k_neighbors")
plt.ylabel("Cohen's kappa")

# make nice plotting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
plt.xlim([1, 10])
plt.ylim([0.4, 0.8])

plt.legend(loc="best")
plt.show()"""



















