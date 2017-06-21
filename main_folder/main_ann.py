# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

print(cleaned_dataset.head())
print(cleaned_dataset.columns)

# FIRST CLEAN

# write to a new Excel file for Tableau Visualization
#cleaned_dataset.to_excel('train.xlsx')

"""# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# One Hot just for the country, but not for Gender (which is binary)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]"""

# Deleting useless columns
cleaned_dataset.drop('SEX', axis=1, inplace=True)
cleaned_dataset.drop('EDUCATION', axis=1, inplace=True)
cleaned_dataset.drop('MARRIAGE', axis=1, inplace=True)
cleaned_dataset.drop('BIRTH_DATE', axis=1, inplace=True)
cleaned_dataset.drop('CUST_COD', axis=1, inplace=True)

print(cleaned_dataset.head())
print(cleaned_dataset.columns)

# Training
X = cleaned_dataset.iloc[:, [0, 18]].values
y = cleaned_dataset.iloc[:, 19].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer 
# Why 6? 11 input + 1 output / 2 - way to select the number of weights
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['f1'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) # 100

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # Baseline?

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['f1'])
    return classifier

# build the classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1, scoring='f1')
mean = accuracies.mean()
variance = accuracies.std()

# Bias Variance tradeoff

print(mean) # Bias
print(variance)

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
from keras.layers import Dropout

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()

    # adding Dropout in case of Overfitting (p = fraction of neurons dropped)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
# parametri per il tuning
# utilizzare il numero di epoche è come fare il tune con early stopping
# ma weight decay?
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# LAST STEP
# after find out wich are the best parameters we can train the model on the entire dataset

print(best_parameters)
print(accuracy)
