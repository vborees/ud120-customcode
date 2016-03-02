#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "*** dataset status ***"
print "number of people:", len(enron_data)
firstperson = enron_data.itervalues().next()
print "number of features", len(firstperson)
print "feature names:"

for key in firstperson:
    print "  ", key

poi_amount = 0
defined_salary = 0
defined_emails = 0
nan_total_payments = 0
poi_nan_total_payments = 0

print "poi list:"
for personkey in enron_data:
    if(enron_data[personkey]["poi"] == 1):
        poi_amount = poi_amount + 1
        if(math.isnan(float(enron_data[personkey]["total_payments"]))):
            poi_nan_total_payments = poi_nan_total_payments + 1
        print "  ", personkey
    if(math.isnan(float(enron_data[personkey]["salary"])) == False):
        defined_salary = defined_salary + 1
    if(enron_data[personkey]["email_address"] != 'NaN'):
        defined_emails = defined_emails + 1
    if(math.isnan(float(enron_data[personkey]["total_payments"]))):
        nan_total_payments = nan_total_payments + 1


print "number of poi's = ", poi_amount

print "stocks of James Prentice", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "number of from to poi emails for Colwell Wesley", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "exercised stocks of Jeffrey Skilling", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "number of persons with defined salary", defined_salary

print "number of persons with defined emails", defined_emails

print "number of persons with nan total payments", nan_total_payments

print "percentage of persons with nan total payments", float(nan_total_payments) / len(enron_data)

print "percentage of poi persons with nan total payments", float(poi_nan_total_payments) / len(enron_data)
