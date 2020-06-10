#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Created on 24 05 2020 - 3:17 PM
  
 @author Sarah Naik
"""
import pandas as pd
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sn
import matplotlib.pyplot as plt

# Read the data
zf = zipfile.ZipFile('../../dataset/Dataset.zip')
features = pd.read_csv(zf.open('Dataset/Questionaire.csv'))

# Labels are the values we want to predict
labels = np.array(features['Classification'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Classification', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#Create model with 100 trees
model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')

# Fit on training data
model.fit(train_features, train_labels)

# Actual class predictions
rf_predictions = model.predict(test_features)

#Accuracy Metrics
print(accuracy_score(test_labels,rf_predictions))
results = confusion_matrix(test_labels, rf_predictions)
print(results)
print(classification_report(test_labels, rf_predictions))

#Accuracy heat map
df_cm = pd.DataFrame(results, index = [i for i in ["Aggressive Investor", "Conservative Investor", "Moderate Investor", "Moderately Aggressive Investor", "Moderately Aggressive Investor"]],
                  columns = [i for i in ["Aggressive Investor", "Conservative Investor", "Moderate Investor", "Moderately Aggressive Investor", "Moderately Aggressive Investor"]])
plt.figure(figsize = (10,8))
sn.heatmap(df_cm, annot=True)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(train_features, train_labels)