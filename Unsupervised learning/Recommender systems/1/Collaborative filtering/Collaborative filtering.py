# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:12:05 2020

@author: lucas
"""

#%% Importing libraries
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#%% Importing Data
movies_df = pd.read_csv('movies.csv')
ratings_df=pd.read_csv('ratings.csv')
movies_df.head()
movies_df.columns
movies_df['title']

#%% Cleaning data
movies_df['title'].str.extract('(\(\d\d\d\d\))',expand = False)
movies_df['year'] = movies_df['title'].str.extract('(\(\d\d\d\d\))',expand = False)
movies_df['year'] = movies_df['year'].str.strip('(\d\d\d\d)')
movies_df['title'] = movies_df['title'].str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].str.strip()
movies_df['genres'] = movies_df['genres'].str.split('|')

ratings_df.columns
ratings_df = ratings_df.drop('timestamp',1)
ratings_df.head()
ratings_df.items
#%%
moviesWithGenres_df = movies_df.copy()
moviesWithGenres_df['genres']
for index,row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index,genre]=1
    
moviesWithGenres_df.columns    
moviesWithGenres_df=moviesWithGenres_df.drop('title',1).drop('genres',1).drop('year',1)    
moviesWithGenres_df.head()    
moviesWithGenres_df=moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()    

#%%
movies_df=movies_df.drop('genres',1)
movies_df.columns
#%% Collaborative Filtering
userInput=[{'title':'Breakfast Club','rating':5},
           {'title':'Toy Story','rating':3.5},
           {'title':'Jumanji','rating':2},
           {'title':'Pulp Fiction','rating':5},
           {'title':'Akira','rating':4.5}
           ]

inputMovies = pd.DataFrame(userInput)
inputMovies
#%% Add movield to input user
#Filtering out the movies by title
inputId=movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title
inputMovies=pd.merge(inputId,inputMovies)
inputMovies
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year',1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation
inputMovies
#%% The users who has seen the same movies
#Filtering out users that have watched movies that the input has watche and storing it
userSubset=ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head(10)
userSubset.shape
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
#lets look at one of the users, e.g the one with userID=1130
userSubsetGroup.get_group(1130)
userSubsetGroup.head()
#len(userSubsetGroup[-1][1])
#Sorting it so user with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,key=lambda x: len(x[1]), reverse = True)
userSubsetGroup[0:3]
#%% SIMILATIRY OF USERS TO INPUT USER
userSubsetGroup = userSubsetGroup[0:100]
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
userSubsetGroup[97]
i=0
#Store the Pearson Correlation ina Dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict ={}
for name, group in userSubsetGroup:
    i = i+1
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]

    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()

    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name]= Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name]=0
    
pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.head()
pearsonDF.columns = ['similarityIndex']    
pearsonDF['userId']= pearsonDF.index
pearsonDF.index = range(len(pearsonDF))    
pearsonDF.head() 

#%%The top x similar users to input user
topUsers=pearsonDF.sort_values(by='similarityIndex',ascending=False)[0:50]    
topUsers.head()    

topUsersRating = topUsers.merge(ratings_df,left_on='userId',right_on='userId',how='inner')    
topUsersRating.head()   

#%% Using Similatiry as weight
#Multiplies the similatiry by user's rating
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()    

#Applies a sum to the topUsers after grouping it by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Creates an empty dataframe
recommendation_df=pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score']= tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId']=tempTopUsersRating.index
recommendation_df.head()

#Now let's sort it and see the top 20 movies that the algorithm recommended
recommendation_df=recommendation_df.sort_values(by='weighted average recommendation score',ascending=False)
recommendation_df.head(10)

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]['title']

#%%
A=[1,2,3,4,5]
B=[2,4,6,8,10]  
A=np.transpose(A) 
B=np.transpose(B) 
test_df = pd.DataFrame([A,B])
test_df
help(pd)


