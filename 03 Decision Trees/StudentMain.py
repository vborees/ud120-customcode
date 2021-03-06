#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../00 data/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import shutil as shutil
from classifyDT import classify
from sklearn import tree

features_train, labels_train, features_test, labels_test = makeTerrainData()

# the classify() function in classifyDT is where the magic happens
clf = classify(features_train, labels_train)

# predict
pred = clf.predict(features_test)

# compute accuracy
acc = clf.score(features_test, labels_test)

print "accuracy for min_sample_split=2:", acc

# build and save the scatter plot to the file
prettyPicture(clf, features_test, labels_test, "test_min_sample_split2.png")
output_image("test_min_sample_split2.png", "png", open("test_min_sample_split2.png", "rb").read())

# get classifier with higher min_sample_split
clf = classify(features_train, labels_train, 50)

# predict
pred = clf.predict(features_test)

# compute accuracy
acc = clf.score(features_test, labels_test)

print "accuracy for min_sample_split=50:", acc

# build and save the scatter plot to the file
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())


