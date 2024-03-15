# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:11:59 2024

@author: Ashish Chincholikar
Random_forest Assignment - 2 
@diabets dataset
"""

"""
1. Business Problem
    for a problem related to healthCare Sector , Diabetes is a condition that can 
    be predicted for any individual based on certain parameters of that person's
    overall health features , we have to develope a model that predicts that wheather
    a person is having the Diabetes or not . this prediction on the features help us 
    enable various analytics data , that a person having similar parameters have more
    chances of developing the diabetes in his near future and hence with this kind of
    insightful data the healthcare experts can ask the individual to take up the neccessary 
    tests to get the diabetes checked and if found positive , immediate medications can 
    be given to the individual 
    
1.1 what is business objective?
    ~To develope a model which predicts the chances of getting diabetes to a person 
    with the given feature or parameters
    ~To identify the Diabetes condition as early as possible and provide the right 
    medication as early as possible
    
1.2 Are there any constraints
    ~Data Collection 
    ~features contributing to diabetes can also be different than the features we are
    assuming and developing model for

2. Create a Data Dictionary 
name of feature 
description 
type 
relevance

1. Pregnancies , No of time the individual has been pregrant , ~ , Relavent data
2. Glucose , Glucose level of that individual , ~ , Relavent data
3. BloodPressure , the blood pressure levels of that indidual , ~  , Relevant data
4. SkinThickness , how thick the skin of that individula is  ,~ , irrelevant
5. Insuline , does the indvisual take insuline , ~ , releavant
6. BMI , BMI of an individual , ~ , relevant
7. Age , Age of that individual , ~ , relevant
8. DiabetesPredigreeFunction , ~ , ~ , irrelevant
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataframe
df = pd.read_csv("C:/Supervised_ML/Random_forest_Algo/Data_Sets/Diabetes.csv")

#print the top records of the Dataframe
df.head

#columns of the dataframe
df.columns

#what are the datatypes of the columns
df.dtypes

#5 number summary of the dataframe
df.describe

# check for null values
df.isnull()


# False
df.isnull().sum()
# 0 no null values



""" 
df.dtypes
pregnant                      int64
Glucose                       int64
BloodPressure                 int64
skinThickness                 int64
Insuline                      int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                      object
dtype: object
"""
##################################################

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)

df.isnull().sum()
df.dropna()
df.columns

# boxplot
# boxplot on Income column
sns.boxplot(df._Number_of_times_pregnant)
# In _Number_of_times_pregnant column 3 outliers 


sns.boxplot(df._Plasma_glucose_concentration)
# In _Plasma_glucose_concentration column 1 outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# histplot - show distributions of datasets
sns.histplot(df['_Number_of_times_pregnant'],kde=True)
# right skew and the distributed

sns.histplot(df['_Plasma_glucose_concentration'],kde=True)
# left skew and the distributed

sns.histplot(df,kde=True)


# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['Outcome'] = enc.fit_transform(df['Outcome'])

#seprate the target variable
y = df.Outcome

X = df.drop(['Outcome'] , axis = "columns")

X.columns

X['pregnant_n'] = enc.fit_transform(X['pregnant'])
X['Glucose_n'] = enc.fit_transform(X['Glucose'])
X['BloodPressure_n'] = enc.fit_transform(X['BloodPressure'])
X['skinThickness_n'] = enc.fit_transform(X['skinThickness'])
X['Insuline_n'] = enc.fit_transform(X['Insuline'])
X['BMI_n'] = enc.fit_transform(X['BMI'])
X['DiabetesPedigreeFunction_n'] = enc.fit_transform(X['DiabetesPedigreeFunction'])
X['Age_n'] = enc.fit_transform(X['Age'])


Xn = X.drop(['pregnant', 'Glucose', 'BloodPressure', 'skinThickness', 'Insuline','BMI', 'DiabetesPedigreeFunction', 'Age'] , axis = 'columns')
y

from sklearn.model_selection import train_test_split
Xn_train , Xn_test , y_train ,y_test = train_test_split(Xn,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
#n_estimators : number of trees in the forest

model.fit(Xn_train , y_train)

model.score(Xn_test , y_test)
y_predicted = model.predict(Xn_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

#%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,7))
sns.heatmap(cm , annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
