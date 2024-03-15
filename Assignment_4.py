# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:27:35 2024

@author: Ashish Chincholikar
Random forest Assignment - 4
@HR_DT.csv DataSet
"""
"""
Business Objective
Minimize: To reduce costs, risks, or inefficiencies in a business process.

Maximize: To increase profits, efficiency, or positive outcomes.
"""
"""
Data Dictionary

 Features                                      Type             Relevance
0   Position of the employee                 Qualititative data  Relevant
1   no of Years of Experience of employee    Continious data     Relevant
2   monthly income of employee               Continious data     Relevant

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/Supervised_ML/Random_forest_Algo/Data_Sets/HR_DT.csv")

df.head(10)
df.tail()
##########################
# 5 number summary
df.describe()
##########################
#shape of the dataset
df.shape
# 600 rows and 6 columns
##########################
df.columns
'''Index(['Position of the employee', 'no of Years of Experience of employee',
       ' monthly income of employee'],
      dtype='object')'''
###########################
# check for null values
df.isnull()
# False
############################
df.isnull().sum()
# no null values
###########################
# Pair-Plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()
# there are the some outlier are present in the dataset

#now we convert into numeric data 
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

df.rename(columns = {' monthly income of employee' : 'Monthly_income_of_employee' , 'Position of the employee' : 'Position_of_the_employee' ,'no of Years of Experience of employee': 'no_of_Years_of_Experience_of_employee'} , inplace = True)

df['Monthly_income_of_employee_n'] = enc.fit_transform(df['Monthly_income_of_employee'])
y = df.Monthly_income_of_employee_n

df['Position_of_the_employee_n'] = enc.fit_transform(df['Position_of_the_employee'])
df['no_of_Years_of_Experience_of_employee_n'] = enc.fit_transform(df['no_of_Years_of_Experience_of_employee'])

df.columns
df.drop(['Position_of_the_employee' , 'Monthly_income_of_employee' , 'no_of_Years_of_Experience_of_employee' ] , axis = "columns")

from sklearn.model_selection import train_test_split
df_train , df_test , y_train ,y_test = train_test_split(df,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
#n_estimators : number of trees in the forest

model.fit(df_train , y_train)

model.score(df_test , y_test)
y_predicted = model.predict(df_test)
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

