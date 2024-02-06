# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:49:09 2024

@author: 91915
"""

"""Random forest algorithm"""

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target'] = iris.target
df[0:5]

X = df.drop('target',axis=1)
Y = df.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model  = RandomForestClassifier(n_estimators=20)
#
model.fit(x_train,y_train)

model.score(x_test,y_test)
y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


#matplotlib
import matplotlib.pyplot as mt
import seaborn as sns
mt.figure(figsize=(12,7))
sns.heatmap(cm,annot=True)
mt.xlabel('Predicted')
mt.ylabel('Truth')
