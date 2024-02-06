# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:31:08 2024

@author: 91915
"""

"""
Divide the diabetes data into train and test datasets and 
build a Random Forest model with Outcome as the output variable. 
"""
import pandas as pd
df = pd.read_csv("C:/1-Python/2-dataset/Diabetes.csv.xls")
"""EDA"""
df.head()
"""
Number of times pregnant  ...   Class variable
0                          6  ...              YES
1                          1  ...               NO
2                          8  ...              YES
3                          1  ...               NO
4                          0  ...              YES"""

df.info()
"""Data columns (total 9 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0    Number of times pregnant      768 non-null    int64  
 1    Plasma glucose concentration  768 non-null    int64  
 2    Diastolic blood pressure      768 non-null    int64  
 3    Triceps skin fold thickness   768 non-null    int64  
 4    2-Hour serum insulin          768 non-null    int64  
 5    Body mass index               768 non-null    float64
 6    Diabetes pedigree function    768 non-null    float64
 7    Age (years)                   768 non-null    int64  
 8    Class variable                768 non-null    object """
 
df.describe()
"""Number of times pregnant  ...   Age (years)
count                 768.000000  ...    768.000000
mean                    3.845052  ...     33.240885
std                     3.369578  ...     11.760232
min                     0.000000  ...     21.000000
25%                     1.000000  ...     24.000000
50%                     3.000000  ...     29.000000
75%                     6.000000  ...     41.000000
max                    17.000000  ...     81.000000"""
#data is widely distributed sp we have to normalize it
#normalization function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#change target column name to outcome
df = df.rename(columns={' Class variable': 'outcome'})

#seperate the input and output columns
X = df.drop('outcome',axis=1)
X = norm_fun(X)
Y = df.outcome

#split the dataset ointo tarining and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#build the random forest classifier
from sklearn.ensemble import RandomForestClassifier
model  = RandomForestClassifier(n_estimators=20)

#train the model
model.fit(x_train,y_train)

#check accuracy score
model.score(x_test,y_test)
"""accuracy score:- 0.7727"""
y_predicted = model.predict(x_test)

#generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
"""
array([[94, 11],
       [24, 25]], dtype=int64)
"""

#generate heatmap of confusion matrix
import matplotlib.pyplot as mt
import seaborn as sns
mt.figure(figsize=(12,7))
sns.heatmap(cm,annot=True)
mt.xlabel('Predicted')
mt.ylabel('Truth')
