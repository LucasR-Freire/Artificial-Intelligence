# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:57:05 2020

@author: lucas
"""
#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pylab as pl

#%%  Load the Cancer data
cell_df = pd.read_csv('cell_samples.csv')
cell_df.head()
cell_df.shape
#df2 = cell_df.drop_duplicates()
print(np.unique(cell_df['Class'].values,axis=0))
cell_df.shape

cell_df['BareNuc']

ax = cell_df[cell_df['Class']== 4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='DarkBlue',label='malignant');
ax2 =cell_df[cell_df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label='benign',ax=ax);
plt.show() 

#%% Data pre-procesing and selection
cell_df.dtypes
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors = 'coerce').notnull()]
cell_df['BareNuc'][50:60]
cell_df['BareNuc']= cell_df['BareNuc'].astype('int')
cell_df.dtypes
#cdf = pd.to_numeric(cell_df['BareNuc'], errors = 'coerce').notnull()
#cdf.loc[cdf.values == False]

feature_df = cell_df[['Clump','UnifSize','MargAdh','SingEpiSize','BareNuc','BlandChrom','NormNucl','Mit']]
X = np.asanyarray(feature_df)
X[0:5]

cell_df['Class']= cell_df['Class'].astype('int')
y = np.asanyarray(cell_df['Class'])
y[0:5]

#%% Train e Test dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#%%Modeling SVM with Scikit-learn
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

yhat = clf.predict(X_test)
yhat[0:5]


#%% Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,
                          normalize = False,
                          title='Confusion matrix', cmap = plt.cm.Blues):
    """
    This  function prints and plots the confusion matrix.
    Normalization can be applied by settings 'normaliza=True'
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis] 
        print("Normalized confusion matrix")
        
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes, rotation =45)
    plt.yticks(tick_marks,classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt),
                 horizontalalignment ='center',
                 color ='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,yhat, labels=[2,4])
print(cnf_matrix)
np.set_printoptions(precision=2)

print(classification_report(y_test,yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Benign(2)','Malignant(4)'],normalize =False, title='Confusion matrix')

#%% Evaluation


#%% Pratice

# print(np.newaxis)
# cnf_matrix = np.array([[85,15],[5,5]])
# cnf_matrix
# cnf_matrix.sum(axis=1)
# cnf_matrix.sum(axis=1)[:,np.newaxis].shape
# cnf_matrix.sum(axis=1)[np.newaxis].shape
# cnf_matrix.astype('float')
# cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:,np.newaxis]
# cnf_matrix.sum(axis=1)
# print(cnf_matrix,'\n\n',cnf_matrix.sum(axis=1)[:,np.newaxis],'\n\n',cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:,np.newaxis])
# cnf_matrix.astype('float') *np.array([2,10])
# np.matmul(cnf_matrix.astype('float'),np.array([2,10]))
# cnf_matrix.astype('float') /[2,10]

# 5./47.
# 5./90.
# 5./52
# (5./85)/52
