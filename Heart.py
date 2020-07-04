# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:48:24 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

df1=pd.read_csv("heart.csv")
df1.isnull().sum()

from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

df1.describe().T

corr=df1.corr()

plt.figure(figsize=[10,7])
sns.heatmap(corr, annot=True)

df1.target.value_counts()

sns.pairplot(df1)

sns.countplot(df1["target"])

nodisease=len(df1[df1.target==0])
hasdisease=len(df1[df1.target==1])

print("% of Patients dont have Heart Disease: {:.2f}%".format((nodisease / (len(df1.target))*100)))
print("% of Patients Have Heart Disease: {:.2f}%".format((hasdisease / (len(df1.target))*100)))

sns.countplot(x='sex',data=df1)
df1.age.value_counts()[:10]

sns.barplot(x=df1.age.value_counts()[:10].index,y=df1.age.value_counts()[:10].values)



pd.crosstab(df1.cp,df1.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency wrt  Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

data=df1.drop(columns=['cp','thal','slope'])

x=data.drop(['target'], axis=1)
y=data['target']

x_data=(x-np.min(x))/(np.max(x)-np.min(x))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

accuracies={}

lr=LogisticRegression()
lr.fit(x_train,y_train)
acc=lr.score(x_test,y_test)*100

accuracies['Logistic Regression']=acc

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))


nb=GaussianNB()
nb.fit(x_train,y_train)
acc=nb.score(x_test,y_test)*100

accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

#K-Nearest Neigbors
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))


#Decision Trees
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
acc = dt.score(x_test, y_test)*100

accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))



ada = AdaBoostClassifier(n_estimators=100)
ada.fit(x_train, y_train)
y_pred = ada.predict(x_test)

accuracies['AdaBoost'] = acc
print("Maximum AdaBoost Score is {:.2f}%".format(acc))

#Random Forest
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)
acc = rf.score(x_test,y_test)*100

accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

xax=list(accuracies.keys())
yax=list(accuracies.values())

#Comparing the Models
x_pos = [i for i, _ in enumerate(xax)]
fig, ax = plt.subplots(figsize=(15,6))

rects1 = ax.bar(x_pos, yax,color=['violet','red','blue','green','orange','cyan'])
plt.xlabel("Models")
plt.ylabel("Accuracy Scores %")
plt.title("Models Comparision")
plt.xticks(x_pos, xax)

def autolabel(rects):
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%f' % float(height),
        ha='center', va='bottom')
autolabel(rects1)
plt.show()



