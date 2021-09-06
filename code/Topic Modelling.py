# The main approach is included here
# The goal is to cluster the papers into a number of clusters and identify the topic of each cluster

# Steps:
# (1) Pre-processes full text (Several steps)
# (2) Calculate TF-iDF score of each keyword in the document collection
# (3) Perform dimension reduction 
# (4) Identify the number of clusters
# (5) Cluster documents
# (6) Extract the keywords representing each cluster

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


#Reading the preprocessed dataset.
#It includes the filtered papers which were selected after preprocessing the metadata file
df = pd.read_csv('D:/Uni_Stuff/complex data project/merged_df1_step2.csv')


#Checking the distribution of the paper releases across the months of the year

#I am going to focus on two different sets.
#Before, and after May!

plt.figure(figsize=(16,8))
plt.hist(df['month_of_publish'],bins=10)
plt.xticks(np.arange(1,12,1))
plt.show()


# Two approaches:
# One on articles before the 5th Month
# One on articles after the 6th month


#Keeping columns
cols = ['paper_id', 'text','title','abstract', 'authors', 'month_of_publish']
df = df[cols]

# Splitting into the two periods

df1 = df[df['month_of_publish'] > 5]
del df1['month_of_publish']

df2 = df[df['month_of_publish'] < 6]
del df2['month_of_publish']

#Full text preprocessing
# STOP WORDS

#Combining scientific with conventional stop words
#Scientific stop words are included on spacy library

#Adding extra Stop_words: These stop words were empiricaly identified
#They include commonly used words on papers and some words refering specificaly to covid

stopwords = list(STOP_WORDS)
extra_stop_words = ['doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table','rights',
'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'cite', 'ade', 'apn'
'Elsevier', 'PMC', 'CZI', 'ct', 'licence' , 'author','doi','study','licence',   '-PRON-', 'usually', 'covid', 'sars','patient'
                   ,'human' ,'coronavirus']
for word in extra_stop_words:
    if word not in stopwords:
        stopwords.append(word)


# TF - iDF !!        
        
        
#Creating a custom tokenizer:
#Defining the pre-processing pipeline according to :
#https://spacy.io/usage/spacy-101#pipelines

model = en_core_sci_lg.load(disable=["tagger", "ner", "parser"])
#Dissabling tagger and ner as I dont care about tagging parts of speach or naming entities
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


#Tokenizing the collections

#For the 1st split of papers
vocabulary = df1['text']
#Fitting
vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized = vectorizer.fit_transform(tqdm(vocabulary))

#For the 2nd split of papers
vocabulary2 = df2['text']
#Fitting
vectorizer2 = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized2 = vectorizer2.fit_transform(tqdm(vocabulary2))

#Vocabularies for the two models has been extracted!
#Converting to arrays and saving for further usage later

array_vectorized = text_vectorized.toarray()
array_vectorized2 = text_vectorized2.toarray()
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'tfidf_database1.txt')
name2 = os.path.join(filepath , 'tfidf_database2.txt')

np.savetxt(name1 , array_vectorized)
np.savetxt(name2, array_vectorized2)

#Dimension Reduction!

#Appling PCA to reduce the dimensions of the TF-iDF matrices.
#In general I look for a balanced reduction so not much information will be lost
#The reduction is empiricaly selected as 90% -> minimum number that the system does not crash
#The primary goal of PCA is to get rid of some noice on the data and sligthly reduce its sparsity
#All of these by keeping as much info as possible

#Defining PCA
pca = PCA(n_components = 0.90, random_state=0) #<- 90% of the total components is enough!
#Applying PCA on the first paper collection!
hope_pca = pca.fit_transform(array_vectorized)
#Applying PCA on the second paper collection!
hope_pca1 = pca.fit_transform(array_vectorized2)

#Saving both for further use
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'pca_database1.txt')
name2 = os.path.join(filepath , 'pca_database2.txt')
np.savetxt(name1 , hope_pca)
np.savetxt(name2, hope_pca1)

# Clustering!


#First of all the most suitable number of clusters needs to be identified
#The elbow method is applied on each paper collection!

#First Set
distortions1 = []
K = range(10,30)

for k in tqdm(K):
    kmeanModel = KMeans(n_clusters = k).fit(hope_pca)
    distortions1.append(sum(np.min(cdist(hope_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / hope_pca.shape[0])
   
#Second Set
distortions2 = []
K = range(10,30)

for k in tqdm(K):
    kmeanModel = KMeans(n_clusters = k).fit(hope_pca1)
    distortions2.append(sum(np.min(cdist(hope_pca1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / hope_pca1.shape[0])
   

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

#14 is selected 


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

#14 as well!!!


#Clustering the first set!
kmeans1 = KMeans(n_clusters = 14 , random_state = 42)
y_pred = kmeans1.fit_predict(hope_pca)
df1['cluster'] = y_pred
#Clustering the second set!
kmeans2 = KMeans(n_clusters = 14 , random_state = 0)
y_pred = kmeans2.fit_predict(hope_pca1)
df2['cluster'] = y_pred



#Adding the papers on their cluster
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


# Topic extraction!! 

#For every cluster, extractινγ the topics discussed on the included papers 

#First creating a vectorizer for every cluster on both documents
#Will use the same tokenizer as defined before

vectorizers1 = []
vectorizers2 = []
for i in range(clusters1): 
    vectorizer = CountVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers1.append(vectorizer)
for i in range(clusters2): 
    vectorizer = CountVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers2.append(vectorizer)


#Running the vectorizers for every cluster on the first collection
vectorized_dfs1 = []
for i in tqdm(range(clusters1)):
    vocabulary1 = clustered_dfs1[i]['text']
    transformed_df = vectorizers1[i].fit_transform(vocabulary1)
    vectorized_dfs1.append(transformed_df)

#Running the vectorizer on the second collection of papers
vectorized_dfs2 = []
for i in tqdm(range(clusters2)):
    vocabulary2 = clustered_dfs2[i]['text']
    transformed_df = vectorizers2[i].fit_transform(vocabulary2)
    vectorized_dfs2.append(transformed_df)




#Get the number of topics for every different type of cluster
#Finding a balanced number of topics by considering three different types of clusters, small, big , medium

#As clusters are imbalanced the aim is to pick a balanced number of topics

#Clusters sizes:
# Big : Over 3000 articles
# Medium: Between 1500 and 3000
# Small :Less than 500

# First set 
# Defining gridsearch
params = {'n_components': [5, 8 ,  10, 12 , 15 ] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[18])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)

#Checking for many papers
params = {'n_components': [5, 8, 10, 12 ,15] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[17])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)

params = {'n_components': [10, 12 ,15 ,17, 20] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[2])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)

#optimal number of topics now:
params = {'n_components': [5, 8 ,  10, 12 , 15 ] }
lda = LatentDirichletAllocation(random_state = 0 , verbose = 1)
gridsearch = GridSearchCV(lda, param_grid = params, n_jobs=-1, verbose=1) #n_jobs is to use all cpus
gridsearch.fit(vectorized_dfs[12])

print("Model's Params: ", gridsearch.best_params_)
print("Log Likelihood Score: ", gridsearch.best_score_)


# According to the results on each type of clusters, a total of 8 topics will be used
# (On small classes 5 had the best score, on medium 7 and on big 12)
# The balanced selection is empiricaly choosen to be 8.

#LDA
#Creating and fittin an lda for every cluster on both collections of documents
ldas1 = []
for i in range(clusters1):
    lda = LatentDirichletAllocation(random_state = 42 , verbose = 0 , n_components = 8, learning_method = 'online')
    ldas1.append(lda)

ldas2 = []
for i in range(clusters2):
    lda = LatentDirichletAllocation(random_state = 0 , verbose = 0 , n_components = 8, learning_method = 'online')
    ldas2.append(lda)


#Fitting on the first collection
fitted_dfs1 = []
for i in tqdm(range(clusters1)):
    fit_df = ldas1[i].fit_transform(vectorized_dfs1[i])
    fitted_dfs1.append(fit_df)
#Fitting on the second collection
fitted_dfs2 = []
for i in tqdm(range(clusters2)):
    fit_df = ldas2[i].fit_transform(vectorized_dfs2[i])
    fitted_dfs2.append(fit_df)


#The following is a function extracting keywords descring a topic within a cluster
#Each cluster has a total of 8 topics
#For every topic, 10 words will extracted and will be added on an array containing the keywords for the cluster
# 10 words for every topic on a cluster
# A total 8 topics means 80 keywords per cluster describing the particular cluster

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

#Creating an array with the keywords for every cluster
keywords1 = [top_keywords('new',cluster) for cluster in range(clusters1)]
#keywords2 = [top_keywords('old',cluster) for cluster in range(clusters2)]

#Saving keywords and df
filepath = 'D:/complex data project'
name1 = os.path.join(filepath , 'keywords111.txt')
name2 = os.path.join(filepath , 'keywords22.txt')
np.savetxt(name1 , keywords1, fmt='%s')
#np.savetxt(name2 , keywords2, fmt='%s')

import pickle
output1 = open('keywords111.pkl', 'wb')
pickle.dump(keywords1, output1)
output1.close()

output2 = open('keywords22.pkl', 'wb')
pickle.dump(keywords2, output2)
output2.close()

pkl_file = open('keywords.pkl', 'rb')
data1 = pickle.load(pkl_file)

filepath = 'D:/complex data project'
nme1 = os.path.join(filepath , 'df_final111.csv')
name2 = os.path.join(filepath , 'df_final22.csv')
df1.to_csv(name1)
df2.to_csv(name2)

