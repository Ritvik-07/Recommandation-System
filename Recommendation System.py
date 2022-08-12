#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv(r'D:\data sets\tmdb_5000_movies.csv')

credits = pd.read_csv(r'D:\data sets\tmdb_5000_credits.csv')


# In[3]:


movies.head(10)


# In[4]:


credits.head(10)


# In[5]:


#merge both the data set on the basis of title

movies = movies.merge(credits, on = 'title')


# In[6]:


movies.shape


# In[7]:


movies.head()


# In[8]:


movies.info()


# In[9]:


movies = movies[['movie_id', 'title','overview',  'genres','keywords',  'cast', 'crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies = movies.dropna()


# In[13]:


movies.isnull().sum()


# # ast — Abstract Syntax Trees
# 

# In[14]:


#


import ast


# In[15]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[16]:


movies['genres'] = movies['genres'].apply(convert)


# In[17]:


movies.head()


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[19]:


movies.head()


# In[20]:


#convert3 function pick the first 3 actors from the cast

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter = counter+1
        else:
            break
    return L
        


# In[21]:


movies.cast = movies.cast.apply(convert3)


# In[22]:


movies.head()


# In[23]:


#function fetch_dicrector will fetch the director name from the crew

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[24]:


movies['crew'][0]


# In[25]:


movies['crew'] = movies.crew.apply(fetch_director)


# In[26]:


movies.head(1)


# In[27]:


movies['overview'] = movies.overview.apply(lambda x : x.split())


# In[28]:


movies.head(1)


# # 
# removing the spaces from geners cast crew etc.
# this is becoz, so our model dont get confused b/w, Sam Worthington and Sam mendes .
# bocoz if we dont do so the model will create two different tags for Sam Worthington one of Sam and other one Worthington and do so for Sam mendes
# so there are two 'sam' tags created which will confuse our model

# In[29]:


movies['genres'] = movies.genres.apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies.keywords.apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies.cast.apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies.crew.apply(lambda x: [i.replace(" ", "") for i in x])


# In[30]:


movies.head(2)


# In[31]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[32]:


movies.head(1)


# In[33]:


cols = ['overview', 'genres', 'keywords', 'cast', 'crew']

df = movies.drop(columns = cols,  axis = 1)


# In[34]:


df.head()


# In[35]:


df['tags'] = df.tags.apply(lambda x : " ".join(x))


# In[36]:


df.head(1)


# In[37]:


df.tags[0]


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[40]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return  ' '.join(y)


# In[41]:


df['tags'] = df.tags.apply(stem)


# In[42]:


df['tags'] = df.tags.apply(lambda x : x.lower())


# In[43]:


df.head(1)


# In[44]:


df.tags[0]


# # Text Vectorization
# 
# We useText vectorization techniques namely Bag of Words.
# 
# we also remove the stop words from our text data
# 

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer


# In[46]:


cv = CountVectorizer( max_features = 5000, stop_words = 'english')

cv.fit_transform(df['tags']).toarray().shape

vectors = cv.fit_transform(df['tags']).toarray()


# In[47]:


vectors[0]


# In[48]:


len(cv.get_feature_names())


# In[49]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[50]:


cv.get_feature_names()


# In[51]:


from sklearn .metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)


# In[52]:


#similarity of first movie from each movie

similarity[0]


# In[53]:


#creating the index value of each movie in 
#just showing the distance b/w 0th movie with other movies in the form of list of tuples

list(enumerate(similarity[0]))


# In[54]:


#now sorting the list with reverse = true, and key = similarity(on the basis of similarity)

sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x : x[1])


# In[55]:


#for fetching the first 5 shortest similarity 

sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x : x[1])[1:6]


# In[56]:


df[df.title == 'Avatar'].index[0]


# In[57]:


#creating a function that recommend us 
'''
def recommend(movie):
    movie_index = df[df.title == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x : x[1])[1:6]
    
    for i in movie_list:
        print(i[0])'''


# In[58]:


recommend('Avatar') #we are getting  index instead of name


# In[59]:


df.iloc[1216].title #in this way we can access the title of the movie


# In[60]:


#recreating the same function
'''
def new_recommend(movie):
    movie_index = df[df.title == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x : x[1])[1:6]
    
    for i in movie_list:
        print(df.iloc[i[0]].title) #changed here
        '''


# In[75]:


def recommend2(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)


# In[77]:


recommend2('Batman Begins') #NOW we are getting the title instead of index


# In[62]:


new_recommend('Batman Begins')


# In[63]:


import pickle


# In[69]:


pickle.dump(df, open('movies.pkl', 'wb'))


# In[70]:


df['title'].values


# In[71]:


pickle.dump(df.to_dict() ,open('movies_dict.pkl', 'wb'))


# In[72]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[ ]:




