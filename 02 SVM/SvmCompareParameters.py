import sys
sys.path.append('../00 data/')


def submitAccuracy():
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


# create classifiers with multiple parameters
from sklearn.svm import SVC
classifiers=[
    SVC(kernel="linear", gamma=1000.0),
    SVC(kernel="linear", gamma=100.0),
    SVC(kernel="linear", gamma=10.0),
    SVC(kernel="linear", gamma=1.0),
    SVC(kernel="linear", gamma=0.1),
    SVC(kernel="linear", gamma=0.01),
    SVC(kernel="linear", gamma=0.001),
    SVC(kernel="rbf", gamma=1000.0),
    SVC(kernel="rbf", gamma=100.0),
    SVC(kernel="rbf", gamma=10.0),
    SVC(kernel="rbf", gamma=1.0),
    SVC(kernel="rbf", gamma=0.1),
    SVC(kernel="rbf", gamma=0.01),
    SVC(kernel="rbf", gamma=0.001)]

filenames=[
    "linear_g1000",
    "linear_g100",
    "linear_g10",
    "linear_g1",
    "linear_g01",
    "linear_g001",
    "linear_g0001",
    "rbf_g1000",
    "rbf_g100",
    "rbf_g10",
    "rbf_g1",
    "rbf_g01",
    "rbf_g001",
    "rbf_g0001",
]

########################## SVM #################################
### we handle the import statement and SVC creation for you here



#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
for i in range(0,len(classifiers)):
    clf = classifiers[i];
    name = filenames[i]

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print name, ' accuracy=', submitAccuracy()

    prettyPicture(clf, features_test, labels_test, name+'.png')