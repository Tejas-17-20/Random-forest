# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:17:51 2024

@author: 91915
"""

"""
A cloth manufacturing company is interested to know about the different attributes 
contributing to high sales. Build a random forest model with Sales as target
variable (first convert it into categorical variable).
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/1-Python/2-dataset/Company_Data.csv.xls")
df

df.head()

df.info()
#dataset does not contains any null values

#we have to convert sales column to categorical data as it is our target column
df['sales_category'] = 'average'
df.loc[df['Sales']<7,'sales_category'] = 'low'
df.loc[df['Sales']>12,'sales_category'] = 'good'

'''df.describe()
#data is widely distriduted so we have to normalize it 
#normalization 
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df)
df_norm'''

df['ShelveLoc'].unique()
df['ShelveLoc'] = pd.factorize(df.ShelveLoc)[0]
df['Urban'] = pd.factorize(df.Urban)[0]
df['US'] = pd.factorize(df.US)[0]
df = df.drop('Sales',axis=1)

X = df.drop('sales_category',axis=1)
Y = df.sales_category

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model  = RandomForestClassifier(n_estimators=20)
model.fit(x_train,y_train)


model.score(x_test,y_test)
"""accuracy of model :- 0.8125"""
y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
"""
array([[27,  0,  6],
       [ 4,  0,  0],
       [ 5,  0, 38]], dtype=int64)"""
