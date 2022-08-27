#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books = pd.read_csv('Books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


ratings.isnull().sum()


# In[10]:


books.duplicated().sum()
books


# In[11]:


users.duplicated().sum()


# In[12]:


ratings.duplicated().sum()
ratings


# Popularity Based Recommender System

# In[13]:


ratings.merge(books,on = "ISBN")


# In[14]:


ratings.merge(books,on = "ISBN").shape


# In[15]:


ratings_with_name = ratings.merge(books,on = 'ISBN')


# In[16]:


ratings_with_name.groupby('Book-Title').count()['Book-Rating']


# In[17]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'}, inplace=True)
num_rating_df


# In[18]:


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)
avg_rating_df


# In[19]:


popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df


# In[20]:


popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating', ascending=False).head(50)


# In[21]:


popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[22]:


popular_df


#  Collaborative Filtering Based Recommender System

# In[23]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
User_Rec=x[x].index
User_Rec


# In[26]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(User_Rec)]


# In[28]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[29]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[32]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[34]:


pt.fillna(0,inplace=True)


# In[35]:


pt


# In[36]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


similarity_scores = cosine_similarity(pt)


# In[41]:


similarity_scores.shape


# In[47]:


def recommend(book_name):
    #index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    for i in similar_items:
        print(pt.index[i[0]])
   


# In[51]:


recommend('A Walk to Remember')


# In[52]:


pt.index[545]


# In[ ]:




