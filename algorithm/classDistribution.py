import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sn


data = pd.read_csv('../dataset/Questionaire.csv')

print(data.shape)

features = data.drop('Classification', axis = 1)
target = np.array(data['Classification'])

counter = Counter(target)

features = np.array(features)

for k,v in counter.items():
    per = v/len(target)*100
    print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42) 

# training a linear SVM classifier
svm_model_linear = SVC(kernel = 'linear', C = 3).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
print(accuracy)
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions)  
print(cm)

print(classification_report(y_test, svm_predictions))

#Accuracy heat map
df_cm = pd.DataFrame(cm, index = [i for i in ["Aggressive Investor", "Conservative Investor", "Moderate Investor", "Moderately Aggressive Investor", "Moderately Aggressive Investor"]],
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
rf = SVC()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
cross_val = cross_val_score( rf, X_train, y_train, cv = 3)
# Fit the random search model
cross_val.fit(X_train, y_train)
cross_val_predictions = cross_val.predict(X_test)

# model accuracy for X_test   
accuracy = cross_val.score(X_test, y_test) 
print(accuracy)
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, cross_val_predictions)  
print(cm)

print(classification_report(y_test, cross_val_predictions))
