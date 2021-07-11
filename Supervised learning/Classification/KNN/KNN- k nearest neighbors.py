
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:27:24 2020

@author: lucas
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#%% Load Data from CSV
df = pd.read_csv('teleCust1000t.csv')
df.head()

#%%Data Visualization and Analysis
df['custcat'].value_counts()
df.income.max()
df.hist(column='income',bins=50)
#%%
df.columns
df.shape
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values #.astype(float)
X[0:5]

y =df['custcat'].values
y[0:5]
set(y)

# Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#%%Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=4)
print( 'Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#%% Classification - K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

#%%Training
k=1
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#%%Predicting
yhat = neigh.predict(X_test)
yhat[0:5]

#%% Accuracy evaluation
from sklearn import metrics
print('Train set Accuracy:', metrics.accuracy_score(y_train,neigh.predict(X_train)))
print('Test set Accuracy:',metrics.accuracy_score(y_test,yhat))


#%% Choosing the best k
from sklearn import metrics
ks = 40
k_list = np.arange(1,ks)
acc_train=[]
acc_test= np.zeros((ks-1))
std_acc = np.zeros((ks-1))
acc_test
std_acc
for k in k_list:
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    acc_test[k-1] = metrics.accuracy_score(y_test,yhat)
    acc2= metrics.accuracy_score(y_train,neigh.predict(X_train))
    acc_train.append(acc2)
    std_acc[k-1]= np.std(yhat== y_test)/np.sqrt(yhat.shape[0])


plt.figure(figsize =(8,6))
plt.plot(k_list,acc_test,'bo-', label ='Test set accuracy')
plt.plot(k_list,acc_train,'r^-', label ='Train set accuracy')
plt.fill_between(k_list,acc_test - 1*std_acc,acc_test + 1*std_acc,alpha=0.10)
plt.title('Measuring the impact of K in accuracy')
plt.xlabel('number of K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig("Impact_of_K_accuracy.png", dpi=300)
#Encontrando o K de accuracy máxima

len(acc_test)
acc_test.argmax()
print( "The best accuracy was with", acc_test.max(), "with k=", acc_test.argmax()+1) 










