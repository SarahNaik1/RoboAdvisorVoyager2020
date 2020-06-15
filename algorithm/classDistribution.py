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