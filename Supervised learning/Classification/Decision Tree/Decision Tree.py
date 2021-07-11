# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:52:35 2020

@author: lucas
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#%% Data
my_data = pd.read_csv('drug200.csv', delimiter=",")
my_data[0:5]
my_data.shape
my_data.describe()
my_data[5:15]
#%% Declare variables
X = my_data[["Age","Sex","BP","Cholesterol",'Na_to_K']].values
y = my_data["Drug"].values
X[:5]
#%% Convert some features to dummies
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[:5]

''' Another example to replace words for numbers
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()'''
#%% SETTING UP THE DECISION TREE
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset ,y_testset = train_test_split(X,y,test_size =0.3,random_state=3)

for i in [X_trainset, X_testset, y_trainset ,y_testset]:
    print(i.shape)
    
#%% MODELING
drugTree = DecisionTreeClassifier(criterion ='entropy',max_depth =6)
drugTree.fit(X_trainset,y_trainset)
#%% Prediction
predTree = drugTree.predict(X_testset)
    
print(predTree [:5])
print(y_testset[:5])

#%% Evaluation
from sklearn import metrics
print("DecisionTree's Accuracy:", metrics.accuracy_score(y_testset,predTree))

#%% Calculate accuracy without sklearn

#%% Visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

#%%
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')