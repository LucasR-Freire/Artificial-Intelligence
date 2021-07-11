# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:21:54 2020

@author: lucas
"""

#%% Preprocessing
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

#%% Read file
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
movies_df.head()
movies_df.columns
movies_df['title']
#%% Cleaning data
#we specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year']= movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year']
#Removing the parantheses
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)', expand=False)
#Removing the years from the 'title' column
movies_df['title']=movies_df.title.str.replace('(\(\d\d\d\d\))','')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'].head()
movies_df['title']=movies_df['title'].apply(lambda x: x.strip())
movies_df['title'].head()

#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'].head()
movies_df['genres']= movies_df['genres'].str.split('|')
movies_df['genres'].head()

#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df= movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index,row in movies_df.iterrows(): 
    for genre in row['genres']:
        moviesWithGenres_df.at[index,genre]=1  
    

# moviesWithGenres_df.head() 
# moviesWithGenres_df.columns
# moviesWithGenres_df['genres'].head()      
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

#%% Ratings Data Cleaning
ratings_df.head()
#Drop removes a specified row or column from a dataframe
ratings_df=ratings_df.drop('timestamp',1)
ratings_df.head()

#%% Content- Baed recommendation system
userInput=[{'title':'Breakfast Club,The', 'rating':5},
           {'title':'Toy Story','rating':3.5},
           {'title':'Jumanji','rating':2},
           {'title':'Pulp Fiction','rating':5},
           {'title':'Akira','rating':4.5}
    ]

inputMovies = pd.DataFrame(userInput)
inputMovies
#%%Add movield to input user
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputId.columns
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
inputMovies.columns
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres',1).drop('year',1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original
#dataframe or it might spelled differently, please check capitalisation.
inputMovies

#Filtering out movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
userMovies
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId',1).drop('title',1).drop('year',1).drop('genres',1)
userGenreTable.shape

#%%
inputMovies['rating']
userGenreTable.transpose()
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userGenreTable.transpose()
#The user profile
userProfile
# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop unnecessary information
genreTable = genreTable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
genreTable.head()
genreTable.shape

#Multiply the genres by the weights and then take the weighted average
recommendationTable_df=((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
recommendationTable_df.shape
recommendationTable_df.columns
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just peak at the values
recommendationTable_df.head()
#The final recommendation table
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
# movies_df[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
# recommendationTable_df.head(20).keys()
# recommendationTable_df.head(20)

#%% Space reserved to practice
# nome = pd.Series(['lucas (1994)','samira 1984'])
# nome.str.extract('(\w \(\d\d\d\d\))',expand=False).dropna()
                 
movies_df['title']
movies_df['title'].apply(lambda x: x.strip())


12.1/1.6




