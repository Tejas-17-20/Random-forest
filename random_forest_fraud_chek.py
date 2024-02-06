# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:59:31 2024

@author: 91915
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('C:/1-Python/2-dataset/Fraud_check.csv.xls')
data
data.columns

# Separate features (X) and target variable (y)
X = data.drop('Urban', axis=1)  # Features
y = data['Urban']  # Target variable
X
y
# One-hot encode the target variable
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
#y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
# One-hot encode the features
X_encoded = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

model.fit(X_train,y_train)
model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


