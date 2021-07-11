# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:38:29 2020

@author: lucas
"""
#%% Libraries
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#%% Generating Data
X1, y1 = make_blobs(n_samples =50, centers =[[4,4],[-2,1],[1,1],[10,4]],cluster_std=0.9)
plt.scatter(X1[:,0],X1[:,1],marker='o')
agglom = AgglomerativeClustering(n_clusters=4,linkage='average')

agglom.fit(X1,y1)

#%%
#Creat figure of size 6 inches by 4 inches
plt.figure(figsize=(6,4))
#These two lines of code are used to scale the data poitns down,
#Or else the data points will scatterd very far apart.

x_min,x_max = np.min(X1,axis=0), np.max(X1,axis=0)
x_min
x_max
# Get the avarage distance for x1
X1 = (X1-x_min)/(x_max-x_min)
#This loog display all of data points.
agglom.labels_
y1
for i in range(X1.shape[0]):
    #Replace the data points with their respective cluster value
    #(ex.0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i,0],X1[i,1],str(y1[i]),
             color = plt.cm.nipy_spectral(agglom.labels_[i] /10),
             fontdict ={'weight': 'bold','size':9})
#Remove xticks, yticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')
#Display the plot of original data before clustering
plt.scatter(X1[:,0],X1[:,1],marker='.')
#Display the plot
plt.show()
#%% Dendogram Associated for the Agglomerative Hierarchical Clustering
dist_matrix=distance_matrix(X1,X1)
print(dist_matrix)
dist_matrix.shape
Z=hierarchy.linkage(dist_matrix,'complete')  
Z.shape
    
dendro = hierarchy.dendrogram(Z)
    
#%% Clustering on Vehicle dataset
#Read
filename = 'cars_clus.csv'
pdf = pd.read_csv(filename)    
print('Shape of dataset:', pdf.shape)    
pdf.head(5)
#%%Cleaning data
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

pdf = pdf.dropna()     #Remove as linhas que possuem algum valor nulo                        
pdf = pdf.reset_index(drop=True) # Adiciona uma novo Índice e remove o antigo
print ('Shape of dataset after cleaning: ',pdf.size)
#%% Feature selection
featureset = pdf[['engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']]

#NORMALIZATION
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx[0:5]
feature_mtx.shape
#%% Clustering using Scipy
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j]=scipy.spatial.distance.euclidean(feature_mtx[i],feature_mtx[j])

#%%
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D,'average')
Z.shape
#%%
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z,max_d,criterion ='distance')
clusters

#%%
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters
#%%
fig = pylab.figure(figsize=(18,50))
def llf(xx):
    return '[%s %s %s]' % (pdf['manufact'][xx], pdf['model'][xx], int(float(pdf['type'][xx])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')


pdf['manufact']
pdf['model']
llf(1)


#%%Clustering using scikit-learn
dist_matrix = distance_matrix(feature_mtx,feature_mtx)
print(dist_matrix)

#
agglom = AgglomerativeClustering(n_clusters =6,linkage='complete')
agglom.fit(feature_mtx)
agglom.labels_
#agglom.labels_.shape
#teste = np.unique(agglom.labels_,return_counts=True)
#teste
#np.sum(teste,axis=1)

pdf['cluster_']= agglom.labels_
pdf.head()
#%%
import matplotlib.cm as cm
n_clusters= max(agglom.labels_)+1
colors =cm.rainbow(np.linspace(0,1,n_clusters))
clusters_labels=list(range(0,n_clusters))

#Create a figure os size 6 inches by 4 inches
plt.figure(figsize=(6,4))

for color , label in zip(colors,clusters_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i],subset.mpg[i],str(subset['model'][i]),rotation=45)
    plt.scatter(subset.horsepow,subset.mpg,s=subset.price*10,c=color,label='cluster'+str(label),alpha=0.5)
    
#plt scatter(subset.horsepow,subset.mpg)
plt.legend()
plt.title('Cluster')
plt.xlabel('horsepow')
plt.ylabel('mpg')

pdf.groupby(['cluster_','type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars

#%%
plt.figure(figsize=(16,10))
# agg_cars.shape
# subset=agg_cars.loc[(1,),]
# subset
# subset.loc[0][0]

for color, label in zip(colors,clusters_labels):
    subset=agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5,subset.loc[i][2],'type='+str(int(i))+', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow,subset.mpg,s=subset.price*20,c=color,label='cluster'+str(label))

plt.legend()
plt.title('Cluster')
plt.xlabel('horsepow')
plt.ylabel('mpg')
    
    
    
    
    
    
    
    
    
    
    












                                 