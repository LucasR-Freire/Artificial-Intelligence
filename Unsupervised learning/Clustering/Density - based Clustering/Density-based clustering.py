# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:24:57 2020

@author: lucas
"""
#%% Importing libraries
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%%
def CreatDataPoints(centroidLocation,numSamples,clusterDeviation):
    # Create random data and store in feature X and response vector y
    X,y = make_blobs(n_samples= numSamples,centers=centroidLocation,cluster_std=clusterDeviation)
    
    #Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X,y
#%% Data
X, y = CreatDataPoints([[4,3],[2,-1],[-1,4]],1500,0.5)
X
#%% Modeling
epsilon=0.3
minimumSamples=7
db=DBSCAN(eps=epsilon,min_samples = minimumSamples).fit(X)
labels=db.labels_
labels.shape
set(labels)

#%% Distinguish outliers
#Firsts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
core_samples_mask[db.core_sample_indices_]=True
core_samples_mask 
#Number of clusters in labels, ignoring noise if present
n_clusters=len(set(labels)) - (1 if -1 in labels else 0)
n_clusters
#Remove repetition in labels by turning it into a set
unique_labels= set(labels)
unique_labels
X.shape
#%% Data visualization 
#Create colors for the clusters
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
#Plot the poitns with colors
labels
for k,col in zip(unique_labels,colors):
    print(k)
    if k==-1:
        #Black used for noise
        col ='k'
    class_member_mask=(labels==k)
    print(type(class_member_mask))
    #Plot the datapoints that are clustered
    xy=X[class_member_mask & core_samples_mask]
    print(xy.shape)
    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker=u'o',alpha=0.5)
    
    #Plot the borders
    xy= X[class_member_mask &~core_samples_mask]
    print(xy.shape)

    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker=u'o',alpha=0.5)
 
#%%
# class_member_mask=(labels==1)
# class_member_mask
# core_samples_mask
# class_member_mask & core_samples_mask
# labels
# False &~False
# False &~ True
# True & True
# True &~ False
# False == True
# class_member_mask &~core_samples_mask
# set(class_member_mask &~core_samples_mask)
# set(core_samples_mask)
# db.core_sample_indices_
# np.array([5,6,7])[np.array([True,True,False]) &~ np.array([True,False,True])]
# [True,True,False] & [True,False,False]
# core_samples_mask
#%%Load the dataset
import pandas as pd
filename = 'weather-stations20140101-20141231.csv'
#Read csv
pdf= pd.read_csv(filename)
pdf.head(5)
pdf.shape
#%% 
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf.shape
pdf = pdf.reset_index(drop=True)
pdf.head(5)

#%% Visualization
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize']=(14,10)

llom = -140
ulon=-50
llat=40
ulat=65

pdf=pdf[(pdf['Long']>llon)&(pdf['Long']<ulon)&(pdf['Lat']>llat)&(pdf['Lat']<ulat)]
my_map = Basemap(projection='Merce',
                 resolution ='1',area_thresh=1000.0,
                 llcrnrlon=llon,llcrnrlat=llat,#min longitude(llcrnrlan) and Latitute(llcrcrLat)
                 ucrcrlon=ulon,ucrcrlat=ulat) #max longitude(ucrnrlon) and latitude(ucrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()

#To collect data based on station

xy,ys = my_map(np.asarray(pdf.Long),np.asarray(pdf.Lat))
pdf['xm']=xs.tolist()
pdf['ym']=ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
    #x,y = my_map(row.Long,row.Lat)
    my_map.plot(row.xm,row.ym,markerfacecolor=([1,0,0]),marker='o',markersize=5,alpha=0.75)
    #plt.text(x,y,stn)
    plt.show()