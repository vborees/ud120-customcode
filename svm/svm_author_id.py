#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from sklearn.svm import SVC
from email_preprocess import preprocess
import numpy as np


def submitAccuracy():
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#timing will be added later

# 0. some preparations
features_train_small = features_train[:len(features_train)/100]
labels_train_small = labels_train[:len(labels_train)/100]

# 1. create classifiers
clf_linear = SVC(kernel="linear")
clf_rbf = SVC(kernel="rbf")
#clf_rbf = SVC(kernel="rbf", C=10000)

clf_linear.fit(features_train, labels_train)
pred = clf_linear.predict(features_test)
print 'linear classifier accuracy=', submitAccuracy()

clf_linear.fit(features_train_small, labels_train_small)
pred = clf_linear.predict(features_test)
print 'linear classifier accuracy, 1% of train dataset used =', submitAccuracy()

clf_linear.fit(features_train_small, labels_train_small)
pred = clf_linear.predict(features_test)
print 'linear classifier accuracy, 1% of train dataset used =', submitAccuracy()

clf_rbf.fit(features_train_small, labels_train_small)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, 1% of train dataset used =', submitAccuracy()

clf_rbf = SVC(kernel="rbf", C=10.0)
clf_rbf.fit(features_train_small, labels_train_small)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, 1% of train dataset used, c=10.0, =', submitAccuracy()

clf_rbf = SVC(kernel="rbf", C=100.0)
clf_rbf.fit(features_train_small, labels_train_small)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, 1% of train dataset used, c=100.0, =', submitAccuracy()

clf_rbf = SVC(kernel="rbf", C=1000.0)
clf_rbf.fit(features_train_small, labels_train_small)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, 1% of train dataset used, c=1000.0, =', submitAccuracy()

clf_rbf = SVC(kernel="rbf", C=10000.0)
clf_rbf.fit(features_train_small, labels_train_small)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, 1% of train dataset used, c=10000.0, =', submitAccuracy()

clf_rbf.fit(features_train, labels_train)
pred = clf_rbf.predict(features_test)
print 'Rbf classifier accuracy, full train dataset used, c=10000.0, =', submitAccuracy()

print 'pred[10] = ',pred[10]
print 'pred[26] = ',pred[26]
print 'pred[50] = ',pred[50]

counts = np.bincount(pred)

print 'Number of classified emails from Chris = ', counts[1]
print 'Number of classified emails from Sara = ', counts[0]


#########################################################


