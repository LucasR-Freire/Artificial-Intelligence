# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:15:36 2020

@author: lucas
"""
#%% Libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

#%% Load Data
churn_df = pd.read_csv('ChurnData.csv')
churn_df.head()
churn_df.shape

#%% Data pre-processing and selection
churn_df = churn_df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()
type(churn_df['age'][0])

#%% Practice
churn_df.shape
X =  np.asanyarray(churn_df[['tenure','age','address','income','ed','employ','equip']])
X[0:5]

y = np.asanyarray(churn_df['churn'])
y[0:5]

#%%  Normalize the dataset
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#%% Train e Test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape,y_test.shape)
#%% Modeling (Logist Regression with Scikit-learn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
yhat_prob

#%% Evaluation
## Jaccard index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,yhat)

## Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix (cm,classes,normalize=False,
                           title='Confusion matrix', cmap=plt.cm.Blues):
    """
      This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True.

    Parameters
    ----------
    cm : TYPE
        DESCRIPTION.
    classes : TYPE
        DESCRIPTION.
    normalize : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is 'Confusion matrix'.
    cmap : TYPE, optional
        DESCRIPTION. The default is plt.cm.Blues.

    Returns
    -------
    None.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm,interpolation ='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks,classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh =cm.max() /2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color ='white' if cm[i,j]>thresh else'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test,yhat,labels=[1,0]))

#Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,yhat,labels=[1,0])
np.set_printoptions(precision=2)


#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['churn=1','churn=01'],
                      normalize=False, title='Confusion matrix')








print(itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])))

cnf_matrix.shape
cnf_matrix


