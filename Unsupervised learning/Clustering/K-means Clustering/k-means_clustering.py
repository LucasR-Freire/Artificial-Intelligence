# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:07:34 2020

@author: lucas
"""

#%% Import libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs

#%% K-Means on a randomly genereted datasets
np.random.seed(0)

X,y = make_blobs(n_samples = 5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]], cluster_std=0.9)
X[0:5,:]
y[0:5]

plt.scatter(X[:,0],X[:,1],marker='.')

#%%Setting up K-means
k_means = KMeans(init = 'k-means++',n_clusters = 3,n_init =12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels

k_means_cluster_centers= k_means.cluster_centers_
k_means_cluster_centers

#%% Creating Visual Plot
# Inicialize the plot with specified dimensions
fig = plt.figure(figsize=(6,4))

# Colors uses a color map, which will produce a n array of colors
# based on the number of labels there are. We use set(k_means_labels)
# to get the unique labels 
colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))
colors
#Creat a plot
ax = fig.add_subplot(1,1,1)
#For loop that plots the data points and centroids
# k will range from 0-3,which will mathe with the possible clusters 
# that each data point is in
for k,col in zip(range(len([[4,4],[-2,-1],[2,-3],[1,1]])), colors):
    #print('valor de k:',k,'\n valor de col:',col,'\n')
    #Creat a list of all data points, where the data points that are
    #in the cluster (ex. cluster 0) are labeled as true, else they are
    #labeled as false
    my_members = (k_means_labels ==k)
    cluster_center = k_means_cluster_centers[k]
    #Plots the data points with color
    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor = col,marker='.')
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor = col,markeredgecolor = 'k',markersize = 6)
ax.set_title('KMeans')
#Remove x-axis ticks
ax.set_xticks(())
ax.set_yticks(())
plt.show()
    
#%% Load Data
cust_df = pd.read_csv('Cust_Segmentation.csv')
cust_df.head()
#Pre Processing
df = cust_df.drop('Address', axis=1)
df.head()
df.loc[1]
df.shape
#Normalizing over the standard deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#%% Modeling
clusterNum = 3
k_means = KMeans(init='k-means++',n_clusters =clusterNum,n_init=12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
#%%Insights
df['Clus_km'] = labels
df.head()  
df.groupby('Clus_km').mean()

legenda = list(set(labels))
legenda
area = np.pi * (X[:,1])**2
plt.scatter(X[:,0],X[:,3],s=area,c=labels.astype(np.float),alpha=0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)

plt.show()

#%% Plot 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1,figsize=(8,6))
plt.clf()
ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:,1],X[:,0],X[:,3], c=labels.astype(np.float))



