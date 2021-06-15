#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import os
from sklearn.manifold import TSNE

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import spacy
import en_core_sci_lg
from spacy.lang.en.stop_words import STOP_WORDS
import re
import seaborn
from tqdm.auto import tqdm

from sklearn.model_selection import GridSearchCV


# In[2]:


metadata = pd.read_csv('D:/complex data project/step1.csv')
del metadata['Unnamed: 0']
#del metadata['month_of_publish']


# In[2]:


#Reading all the papers!!!
df1 = pd.read_csv('D:/complex data project/paper1.csv')
df2 = pd.read_csv('D:/complex data project/paper2.csv')
df3 = pd.read_csv('D:/complex data project/paper3.csv')
df4 = pd.read_csv('D:/complex data project/paper4.csv')
df5 = pd.read_csv('D:/complex data project/paper5.csv')

df6 = pd.read_csv('D:/complex data project/paper_round2_1.csv')
df7 = pd.read_csv('D:/complex data project/paper_round2_2.csv')
df8 = pd.read_csv('D:/complex data project/paper_round2_3.csv')
df9 = pd.read_csv('D:/complex data project/paper_round2_4.csv')
df10 = pd.read_csv('D:/complex data project/paper_round2_5.csv')


# In[7]:


#Trying to merge:
#I will maybe need the rest of the papers(from the other source)

#First taking only the papers which are included on the main source
new_dfs = []
set_of_dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
for df in set_of_dfs:
    df = pd.merge(df, metadata, how = "inner" , left_on ="paper_id" , right_on = "sha")
    new_dfs.append(df)


# In[8]:


df = pd.concat(new_dfs).drop_duplicates()
df


# In[3]:


#df['month_of_publish'].unique()


# In[22]:


#filepath = 'D:/complex data project'
#name = os.path.join(filepath , 'merged_df1_step2.csv')
#df.to_csv(name)


# In[3]:


df = pd.read_csv('D:/Uni_Stuff/complex data project/merged_df1_step2.csv')


# In[6]:


#Taking the months as pointed out on the previous frame!
#df['publish_time'] = pd.to_datetime(df['publish_time'])
#check_dates = [df.iloc[i]['publish_time'].month for i in range(len(df))]


# In[7]:


#The months of the articles!
#I am going to focus on two different sets.
#Before, and after May!

plt.figure(figsize=(16,8))
plt.hist(df['month_of_publish'],bins=10)
plt.xticks(np.arange(1,12,1))
plt.show()


# ### The bellow steps wont be followed as the approach was not followed

# In[8]:


#Grouping by cluster:

#clustered_dfs = []
#clusters = len(df['cluster'].unique())

#for i in range(clusters): 
#    new_df = df[df['cluster'] == i]
#    clustered_dfs.append(new_df)


# In[9]:


#Checking the lenght of each dataframe:

#lenghts = [len(df) for df in clustered_dfs]

#plt.figure(figsize = (14,7))
#plt.barh(np.arange(0,len(lenghts),1) , lenghts )
#plt.title("Papers contained on each cluster")
#plt.xlabel("Number of Papers")
#plt.ylabel("Cluster")
#plt.yticks(np.arange(0,19,1))
#plt.xticks(np.arange(0,3500,50))


# In[ ]:


#Outbalanced classes
#Perform topic modelling!

#Will create a vectorizer for every cluster
#The tokenizer will be similar to the one i used before


# In[11]:


#stopwords = list(STOP_WORDS)
#model = en_core_sci_lg.load(disable=["tagger", "ner"])
#Dissabling tagger and ner as I dont care about tagging parts of speach or naming entities

#dont know about the parser if will include

#Size of the array -for tokenizing
#model.max_length = 3000000
#def custom_tokenizer(sentence):
    #Removing all punchations with a space
 #   sentence =re.sub(r'[^\w\s]',' ', sentence)
    #Applying the pipeline
  #  sentence = model(sentence)
    #Lematizing and lowercasing
    #Removing stopwords, number, punchations, spaces and words with a single letter
   # tokens = [word.lemma_.lower() for word in sentence if ((str(word) not in stopwords) and (word.is_punct==False) and 
    #          (len(str(word))!=1) and (word.like_num==False) and (word.is_space==False))]
    #return(tokens)


# In[12]:


#For each cluster creating a vectorizer:

#vectorizers = []

#for i in range(clusters): 
#    vectorizer = CountVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
#    vectorizers.append(vectorizer)


# In[13]:


#Vectorize data on each cluster

#vectorized_dfs = []
#for i in tqdm(range(clusters)):
#    vocabulary = clustered_dfs[i]['text']
#    transformed_df = vectorizers[i].fit_transform(tqdm(vocabulary))
#    vectorized_dfs.append(transformed_df)


# In[14]:


#As clusters are outbalanced will try to find the optimal number of topics for three sets of clusters:
#Over 3000
#Between 1500 and 3000
#Less than 500

#First set 
# on the [10, 12 ,15 ,17, 20] it picked 10, lowering the values
#params = {'n_components': [5, 8 , 9, 10, 12  ] }
#lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
#gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
#gridsearch.fit(vectorized_dfs[18])

#print("Model's Params: ", gridsearch.best_params_)
#print("Log Likelihood Score: ", gridsearch.best_score_)


# In[15]:


#Best was 5 for very low number of topics
#Checking for many papers
#params = {'n_components': [10, 12 ,15 ,17, 20] }
#lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
#gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
#gridsearch.fit(vectorized_dfs[17])

#print("Model's Params: ", gridsearch.best_params_)
#print("Log Likelihood Score: ", gridsearch.best_score_)


# In[16]:


#12 for the long one:
#For middle one:
#params = {'n_components': [10, 12 ,15 ,17, 20] }
#lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
#gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
#gridsearch.fit(vectorized_dfs[2])

#print("Model's Params: ", gridsearch.best_params_)
#print("Log Likelihood Score: ", gridsearch.best_score_)


# In[ ]:


#To summarize, smaller datasets point that 5 topics should be picked, medium state 10 and max state 12.
#My decisions is 10 topics per cluster... If results are not good enough, will reconsider


# In[17]:


#Defining a model for every cluster

#ldas = []
#for i in range(clusters):
#    lda = LatentDirichletAllocation(random_state = 0 , verbose = 0 , n_components = 10, learning_method = 'online')
#    ldas.append(lda)

#fitted_dfs = []
#for i in tqdm(range(clusters)):
#    fit_df = ldas[i].fit_transform(vectorized_dfs[i])
#    fitted_dfs.append(fit_df)


# In[18]:


#Fitting each cluster:
#fitted_dfs = []
#for i in tqdm(range(clusters)):
#    fit_df = ldas[i].fit_transform(vectorized_dfs[i])
#    fitted_dfs.append(fit_df)


# In[19]:


#Print keywords for each cluster:
#Each cluster covers a number of topics and each topic has a number of keywords describing it
#For example the topics of cluster 0 are described by the following keywords

#def get_keywords(cluster, number_of_keywords = 5 ):
 
#for index, topic in enumerate(ldas[0].components_):
   
    #print([vectorizers[18].get_feature_names()[i] for i in topic.argsort()[-10:]])
    #print('\n')
#    test = [vectorizers[18].get_feature_names()[i] for i in topic.argsort()[-10:]]
#    for word in test: print(word)
#    print("--")


# In[ ]:


#Remove et, al, study, cite, ade , apn , copyright, abbreviation, ct , aforementioned, dot,licence,
#research, author, doi
#include some more from kaggle
#Try another approach with clustering on the whole set of values and then LDA


#State that the method with abstracts failed but apply methods on the whole set(on the merged df1)

#Skip the t-sne visuliazition


# In[20]:


#function for extracting top words:

#def top_keywords(cluster , number_of_words = 5 ):
#    words_to_keep = []
#    for index, topic in enumerate(ldas[cluster].components_):
#        words = [vectorizers[cluster].get_feature_names()[i] for i in topic.argsort()[-number_of_words:]]
#        for word in words:
#            if word not in words_to_keep:
#                words_to_keep.append(word)

#    return(words_to_keep)
        
    


# ### Here I perform all the approaches on the whole dataset!!!
# 
# Two approaches:
# 
# One on articles after the 5th Month
# 
# One on articles from before the 6 month
# 
# * Maybe the initial search will be about which set of papers has the highest TOTAL Tf-IDF on the keywords
# * So save the Tf-IDF vector!!
# 

# In[22]:


#Old One
#df1 = pd.read_csv('D:/complex data project/final_df1.csv')
#df1


# In[26]:


#df.columns
cols = ['paper_id', 'text','title','abstract', 'authors', 'month_of_publish']
df = df[cols]


# ### Approach for papers after May

# In[9]:


df1 = df[df['month_of_publish'] > 5]
del df1['month_of_publish']


# In[10]:


df2 = df[df['month_of_publish'] < 6]
del df2['month_of_publish']


# In[3]:


#df['publish_time'] = pd.to_datetime(df['publish_time'])
#check_dates = [df.iloc[i]['publish_time'].month for i in range(len(df))]

#def new_articles(date):
#    if date.month > 5:
#        return(True)
#    else:
#        return(False)
#df['publish_time'] = pd.to_datetime(df['publish_time'])
#df['to_keep'] = df.apply(lambda row: new_articles(row['publish_time']),axis = 1 )
#df = df[df['to_keep']==True]
#del df["to_keep"]


# In[19]:


#STOP WORDS

#I am going to combine scientific with normal stop words
#Scientific stop words are included on spacy library and they accompany the model which I have defined!


#Adding some extra Stop_words: These stop words were identified on my initial approach
#They include commonly used words on papers and some words refering specificaly to covid


stopwords = list(STOP_WORDS)
extra_stop_words = ['doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table','rights',
'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'cite', 'ade', 'apn'
'Elsevier', 'PMC', 'CZI', 'ct', 'licence' , 'author','doi','study','licence',   '-PRON-', 'usually', 'covid', 'sars','patient'
                   ,'human' ,'coronavirus']
for word in extra_stop_words:
    if word not in stopwords:
        stopwords.append(word)


# In[16]:


#Creating the tokenizer:

#Defining the pre-processing pipeline according to :
#https://spacy.io/usage/spacy-101#pipelines

model = en_core_sci_lg.load(disable=["tagger", "ner", "parser"])
#Dissabling tagger and ner as I dont care about tagging parts of speach or naming entities

#dont know about the parser if will include
#Size of the array -for tokenizing
model.max_length = 3000000
def custom_tokenizer(sentence):
    #Removing all punchations with a space
    sentence =re.sub(r'[^\w\s]',' ', sentence)
    #Splitting some numbers and words so to be removed by the tokenizer: for example 7ac
    sentence = " ".join(re.split(r'(\d+)', sentence))
    #Applying the pipeline
    sentence = model(sentence)
    #Lematizing and lowercasing
    #Removing stopwords, number, punchations, spaces and words with a single letter
    tokens = [word.lemma_.lower() for word in sentence if ((str(word) not in stopwords) and (word.is_punct==False) and 
              (len(str(word))!=1) and (word.like_num==False) and (word.is_space==False))]
    return(tokens)


# In[32]:


vocabulary = df1['text']
#Fitting
vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized = vectorizer.fit_transform(tqdm(vocabulary))


# In[35]:


vocabulary2 = df2['text']
#Fitting
vectorizer2 = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized2 = vectorizer2.fit_transform(tqdm(vocabulary2))


# In[36]:


#Vocabulary of the two models has been extracted!
#Converting to arrays and saving for further usage later

array_vectorized = text_vectorized.toarray()
array_vectorized2 = text_vectorized2.toarray()


# In[ ]:


import pickle
output1 = open('keywords1.pkl', 'wb')
pickle.dump(keywords1, output1)
output1.close()

output2 = open('keywords2.pkl', 'wb')
pickle.dump(keywords2, output2)
output2.close()


# In[38]:


#Saving
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'tfidf_database1.txt')
name2 = os.path.join(filepath , 'tfidf_database2.txt')

np.savetxt(name1 , array_vectorized)
np.savetxt(name2, array_vectorized2)


# In[40]:


#Dimension Reduction!

#Appling PCA to reduce the dimensions of the BOW matrices.
#In general not a big reduction is necessary so not much information will be lost
#Arround 90% of the total matrices will be enough
#The primary goal of PCA is to get rid of some noices on the data and sligthly reduce the sparsity
#All of these by keeping as much info as possible

#Defining PCA
pca = PCA(n_components = 0.90, random_state=0) #<- 90% of the total components is enough!


# In[41]:


#Applying PCA on the first paper collection!
hope_pca = pca.fit_transform(array_vectorized)


# In[42]:


#Applying PCA on the second paper collection!
hope_pca1 = pca.fit_transform(array_vectorized2)


# In[43]:


#Saving both pcas for further use
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'pca_database1.txt')
name2 = os.path.join(filepath , 'pca_database2.txt')

np.savetxt(name1 , hope_pca)
np.savetxt(name2, hope_pca1)


# In[3]:


filepath = 'D:/complex data project'
name = os.path.join(filepath , 'pca_database1.txt')
hope_pca = np.loadtxt(name)
name = os.path.join(filepath , 'pca_database2.txt')

hope_pca1 = np.loadtxt(name)


# In[4]:


#Saving array of pca
filepath = 'D:/complex data project'
name = os.path.join(filepath , 'pcaed_docs.txt')
hope_pca = np.loadtxt(name)


# In[ ]:


#The next step is clustering!
#First of all the most suitable number of clusters should be identified

#To do that the elbow method will be applied on each paper collection!


# In[45]:


#First Set
distortions1 = []
K = range(10,30)

for k in tqdm(K):
    kmeanModel = KMeans(n_clusters = k).fit(hope_pca)
    distortions1.append(sum(np.min(cdist(hope_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / hope_pca.shape[0])
   


# In[46]:


#Second Set
distortions2 = []
K = range(10,30)

for k in tqdm(K):
    kmeanModel = KMeans(n_clusters = k).fit(hope_pca1)
    distortions2.append(sum(np.min(cdist(hope_pca1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / hope_pca1.shape[0])
   


# In[50]:


#Plotting the results

X = np.arange(10,30)
K = range(10,30)
plt.figure(figsize = (16,8))
plt.plot(K,distortions1 , color = 'red' ,marker= 'o', linewidth = 2)
plt.xticks(np.arange(15,35,5))
plt.vlines(13, max(distortions1) , min(distortions1) , linestyle = '--')
plt.vlines(14, max(distortions1) , min(distortions1) , linestyle = '--')

plt.xlabel('Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()


# In[7]:


#14 will be selected as I believe is the best number for papers clusters and by considering the elbow method
#Maybe 13???????????? 
#Second plot


# In[51]:


X = np.arange(10,30)
K = range(10,30)
plt.figure(figsize = (16,8))
plt.plot(K,distortions2 , color = 'red' ,marker= 'o', linewidth = 2)
plt.xticks(np.arange(15,35,5))
plt.vlines(13, max(distortions2) , min(distortions2) , linestyle = '--')
plt.vlines(14, max(distortions2) , min(distortions2) , linestyle = '--')

plt.xlabel('Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()


# In[ ]:


#14 as well!!!


# In[42]:


#Clustering the first set!
#Have added different seed number first 0 then 42 then 32
kmeans1 = KMeans(n_clusters = 14 , random_state = 42)
y_pred = kmeans1.fit_predict(hope_pca)
df1['cluster'] = y_pred


# In[37]:


#Clustering the second set!
kmeans2 = KMeans(n_clusters = 14 , random_state = 0)
y_pred = kmeans2.fit_predict(hope_pca1)
df2['cluster'] = y_pred


# In[43]:


#Grouping the papers on each of their cluster
clustered_dfs1 = []
clusters1 = len(df1['cluster'].unique())

for i in range(clusters1): 
    new_df = df1[df1['cluster'] == i]
    clustered_dfs1.append(new_df)

clustered_dfs2 = []
clusters2 = len(df2['cluster'].unique())

for i in range(clusters2): 
    new_df = df2[df2['cluster'] == i]
    clustered_dfs2.append(new_df)


# In[ ]:


#For every cluster, will extract the topics which are discussed on the papers included
#To do that I have to define a vectorizer for every cluster on both paper collections
#Will use the same tokenizer as defined before


# In[44]:


vectorizers1 = []
vectorizers2 = []
for i in range(clusters1): 
    vectorizer = CountVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers1.append(vectorizer)
for i in range(clusters2): 
    vectorizer = CountVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers2.append(vectorizer)


# In[45]:


#Running the vectorizers for every cluster on the first collection
vectorized_dfs1 = []
for i in tqdm(range(clusters1)):
    vocabulary1 = clustered_dfs1[i]['text']
    transformed_df = vectorizers1[i].fit_transform(vocabulary1)
    vectorized_dfs1.append(transformed_df)


# In[22]:


#Running the vectorizer on the second collection of papers
vectorized_dfs2 = []
for i in tqdm(range(clusters2)):
    vocabulary2 = clustered_dfs2[i]['text']
    transformed_df = vectorizers2[i].fit_transform(vocabulary2)
    vectorized_dfs2.append(transformed_df)


# In[ ]:


#Find an accurate number of topics for every different cluster
#Will find optimla topics for three different kind of clusters, small, big , medium
#Will decide to find a size of topics which will satisfy every cluster(a middle solution)


# In[ ]:


#Going for 8 topics, state sth or run it again


# In[25]:


#As clusters are outbalanced will try to find the optimal number of topics for three sets of clusters:
#Over 3000
#Between 1500 and 3000
#Less than 500

#First set 
# on the [10, 12 ,15 ,17, 20] it picked 10, lowering the values
params = {'n_components': [5, 8 ,  10, 12 , 15 ] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[18])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)


# In[27]:


#Best was 5 for very low number of topics
#Checking for many papers
params = {'n_components': [5, 8, 10, 12 ,15] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[17])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)


# In[ ]:


params = {'n_components': [10, 12 ,15 ,17, 20] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[2])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)


# In[35]:


#optimal number of topics now:
params = {'n_components': [5, 8 ,  10, 12 , 15 ] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[12])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)


# In[ ]:


# A total of 8 topics will be selected as the midium picked 5.
#Creating the ldas for every cluster on both paper collections


# In[48]:


#Creating and fitting the ldas
#First random state 0 then 42 then 32
ldas1 = []
for i in range(clusters1):
    lda = LatentDirichletAllocation(random_state = 42 , verbose = 0 , n_components = 8, learning_method = 'online')
    ldas1.append(lda)

ldas2 = []
for i in range(clusters2):
    lda = LatentDirichletAllocation(random_state = 0 , verbose = 0 , n_components = 8, learning_method = 'online')
    ldas2.append(lda)


# In[49]:


#Fitting on the first collection
fitted_dfs1 = []
for i in tqdm(range(clusters1)):
    fit_df = ldas1[i].fit_transform(vectorized_dfs1[i])
    fitted_dfs1.append(fit_df)


# In[25]:


#Fitting on the second collection
fitted_dfs2 = []
for i in tqdm(range(clusters2)):
    fit_df = ldas2[i].fit_transform(vectorized_dfs2[i])
    fitted_dfs2.append(fit_df)


# In[50]:


#A function which extracts the keywords talking for a speicif topic within a clusters
#Each cluster has a total of 8 topics
#For every topic, 10 words will extracted and will be added on an array containing all the keywords for cluster

#Extracting the keywords for every cluster -> 10 words for every topic!
#So a total of 80 words. Out of these 80 words some are included on many topics so adding them only once!

def top_keywords(collection,cluster, number_of_words = 10 ): 
    #collection is either old or new with old being the second and new the first
    #cluster is given on the functions call
    if collection == 'new':
        ldas = ldas1
        vectorizers = vectorizers1
    else:
        ldas = ldas2
        vectorizers = vectorizers2
    words_to_keep = []
    for index, topic in enumerate(ldas[cluster].components_):
        words = [vectorizers[cluster].get_feature_names()[i] for i in topic.argsort()[-number_of_words:]]
        for word in words:
            if word not in words_to_keep:
                words_to_keep.append(word)

    return(words_to_keep)



# In[51]:


#Creating an array with the keywords for every cluster
keywords1 = [top_keywords('new',cluster) for cluster in range(clusters1)]
#keywords2 = [top_keywords('old',cluster) for cluster in range(clusters2)]


# In[52]:


#Saving keywords and df
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'keywords111.txt')
name2 = os.path.join(filepath , 'keywords22.txt')
np.savetxt(name1 , keywords1, fmt='%s')
#np.savetxt(name2 , keywords2, fmt='%s')


# In[53]:


import pickle
output1 = open('keywords111.pkl', 'wb')
pickle.dump(keywords1, output1)
output1.close()

output2 = open('keywords22.pkl', 'wb')
pickle.dump(keywords2, output2)
output2.close()


# In[59]:


pkl_file = open('keywords.pkl', 'rb')
data1 = pickle.load(pkl_file)


# In[55]:


filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'df_final111.csv')
name2 = os.path.join(filepath , 'df_final22.csv')
df1.to_csv(name1)
df2.to_csv(name2)

