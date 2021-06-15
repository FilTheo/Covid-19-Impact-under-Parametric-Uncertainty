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

from tqdm.auto import tqdm

from sklearn.model_selection import GridSearchCV
import pickle
import functools


# In[2]:


#Reading every file
D:\Uni_Stuff\COVID-19


# In[9]:


df1 = pd.read_csv('D://Uni_Stuff/complex data project/df_final111.csv')
df2 = pd.read_csv('D:/Uni_Stuff/complex data project/df_final22.csv')
pkl_file = open('D:/Uni_Stuff/COVID-19/keywords111.pkl', 'rb')
keywords1 = pickle.load(pkl_file)
pkl_file = open('D:/Uni_Stuff/COVID-19/keywords22.pkl', 'rb')
keywords2 = pickle.load(pkl_file)
#cols = df.columns[3:]
#df = df[cols]


# In[10]:


pkl_file = open('D:/Uni_Stuff/COVID-19/keywords111.pkl', 'rb')
keywords1 = pickle.load(pkl_file)

pkl_file = open('D:/Uni_Stuff/COVID-19/keywords1.pkl', 'rb')
keywords11 = pickle.load(pkl_file)


# In[11]:


keywords1 == keywords11


# In[14]:


#Create a tf-idf vectorizer for every cluster and add the sums
#Splitting dfs based on their cluster
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


#Will get the tf-idf score of both databases and from every cluster within them both


# In[ ]:


stopwords


# In[15]:


#Defining the vectorizer 
stopwords = list(STOP_WORDS)
extra_stop_words = ['doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table','rights',
'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'cite', 'ade', 'apn'
'Elsevier', 'PMC', 'CZI', 'ct', 'licence' , 'author','doi','study','licence',   '-PRON-', 'usually', 'covid', 'sars','patient'
                   ,'human' ,'coronavirus']
for word in extra_stop_words:
    if word not in stopwords:
        stopwords.append(word)

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


# In[16]:


#Creating a vectorizer for every dataset collection
#Will get the tf-idf score again as it was time-consuming loading it
vocabulary = df1['text']
#Fitting
vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized1 = vectorizer.fit_transform(tqdm(vocabulary))


# In[17]:


#Moving to the second paper collection
vocabulary = df2['text']
#Fitting
vectorizer1 = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized2 = vectorizer1.fit_transform(tqdm(vocabulary))


# In[18]:


#Getting the tf-idf scores for every term on the two collections
vectorized_df1 = pd.DataFrame(columns = vectorizer.get_feature_names() , data = text_vectorized1.toarray())
sums = np.array([np.sum(vectorized_df1[j]) for j in vectorized_df1.columns])
tf_idf_scores1 = pd.DataFrame(index = vectorized_df1.columns , data = sums).rename(columns = {0:'Tf-IDF'})
tf_idf_scores1 = tf_idf_scores1.sort_values('Tf-IDF' , ascending = False)
tf_idf_scores1 = tf_idf_scores1.reset_index().rename(columns = {'index':'token'})


# In[19]:


#Getting the tf-idf scores for every term on the two collections
vectorized_df2 = pd.DataFrame(columns = vectorizer1.get_feature_names() , data = text_vectorized2.toarray())
sums = np.array([np.sum(vectorized_df2[j]) for j in vectorized_df2.columns])
tf_idf_scores2 = pd.DataFrame(index = vectorized_df2.columns , data = sums).rename(columns = {0:'Tf-IDF'})
tf_idf_scores2 = tf_idf_scores2.sort_values('Tf-IDF' , ascending = False)
tf_idf_scores2 = tf_idf_scores2.reset_index().rename(columns = {'index':'token'})


# In[20]:


#Creating the vectorizers
vectorizers1 = []
#vectorizers2 = []
for i in range(clusters1): 
    vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers1.append(vectorizer)
    
#Keeping the second for the DDDM only
#for i in range(clusters2): 
 #   vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
  #  vectorizers2.append(vectorizer)


# In[21]:


#Now getting the tf-ifd of each dataframe clustered within both database papers

tf_idf_df1 = []

for i in tqdm(range(clusters1)):
    vocabulary = clustered_dfs1[i]['text']
    transformed_df = vectorizers1[i].fit_transform(vocabulary)
    vectorized_df = pd.DataFrame(columns = vectorizers1[i].get_feature_names() , data = transformed_df.toarray())
    sums = np.array([np.sum(vectorized_df[j]) for j in vectorized_df.columns])
    tf_idf_scores = pd.DataFrame(index = vectorized_df.columns , data = sums).rename(columns = {0:'Tf-IDF'})
    tf_idf_scores = tf_idf_scores.sort_values('Tf-IDF' , ascending = False)
    tf_idf_scores = tf_idf_scores.reset_index().rename(columns = {'index':'token'})

    tf_idf_df1.append(tf_idf_scores)


# In[ ]:


tf_idf_df2 = []

for i in tqdm(range(clusters2)):
    vocabulary = clustered_dfs2[i]['text']
    transformed_df = vectorizers2[i].fit_transform(vocabulary)
    vectorized_df = pd.DataFrame(columns = vectorizers2[i].get_feature_names() , data = transformed_df.toarray())
    sums = np.array([np.sum(vectorized_df[j]) for j in vectorized_df.columns])
    tf_idf_scores = pd.DataFrame(index = vectorized_df.columns , data = sums).rename(columns = {0:'Tf-IDF'})
    tf_idf_scores = tf_idf_scores.sort_values('Tf-IDF' , ascending = False)
    tf_idf_scores = tf_idf_scores.reset_index().rename(columns = {'index':'token'})

    tf_idf_df2.append(tf_idf_scores)


# In[7]:


def index_finder(word,cluster,collection):
    #Setting a stopping element
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    idx = None
    for j in range(len(tf_idf_df[cluster])):
        if tf_idf_df[cluster].iloc[j]['token'] == word:
            idx = j
    return(idx)
def topic_keywords(words,current_selected,collection):
    extra = []
    if collection == 'new':
        keywords = keywords1
        clusters = clusters1
    else:
        keywords = keywords2
        clusters = clusters2
    #words = sentence.split(' ')
    for cluster in range(clusters):
        for word in words:
            #print(word)
            if word in keywords[cluster]:
                extra.append(cluster)
    #print(extra)
    for new_cluster in extra:
        if new_cluster not in current_selected:
            #print(new_cluster)
            current_selected = np.append(current_selected,new_cluster)
    return(current_selected)
 
def clusters_including(query, number_of_clusters, collection):    
    #Returns the tf-idf scores on every cluster.
    #Mean value if more than two words are given
    #number_of_clusters is the number of clusters to return, if not enough docs are found, then i can increase the number
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    words = query.split(' ')
    tf_scores = [np.mean([tf_idf_df[cluster].iloc[index_finder(word,cluster,collection)]['Tf-IDF'] 
                 if index_finder(word,cluster,collection)!= None else 0 for word in words]) for cluster in range(clusters) ]
    #returns two best clusters
    selected_clusters = np.argsort(tf_scores)[::-1][:number_of_clusters]
    #Checking if either word is included on the topics of each cluster extracted by the LDA
    #If so they will also be included
    extra = topic_keywords(words, selected_clusters, collection)
    return(extra)

def get_merged_df(clusters,collection):
#Returns the df which is the merge of the dfs concering each picked cluster
#Input is the output of clusters_including
    if collection == 'new':
        clustered_dfs = clustered_dfs1
        #clusters = clusters1
    else:
        clustered_dfs = clustered_dfs2
        #clusters = clusters2
    picked_dfs = [clustered_dfs[i] for i in clusters]
    merged_df = pd.concat(picked_dfs)
    return(merged_df)
def search_abstracts(sentence, df_to_search):
    #Returns a df where the keywords are included in the abstracts 
    words = sentence.split(' ')
    df_reduced = df_to_search [functools.reduce(lambda a, b: a & b, 
                                                (df_to_search['abstract'].str.contains(word) for word in words))]
    return(df_reduced)
#Here i can define a different search scope, for example incubation, days , range 
#On the pattern search(previous step) i am looking for more abstract searches while here for more detailed ones:
#For example for incubation i can add days as well!!!
#Check the title of the article if it is the same
#Add a formula for pointing if numbers are wanted or not

def get_single_sentence(keywords, text, search_numbers=True):
    #Text is the text of a single paper
    #Splitting text into sentences
    sentences = text.split('. ')
    #search keywords and number (parameter for that)
    if search_numbers == True:
        relevant = [sentence  for sentence in sentences if (all(map(lambda word: word in sentence.lower(), keywords))) 
                    and len(re.findall(r'\d{1,2}',sentence)) ]
    else:
        relevant = [sentence  for sentence in sentences if (all(map(lambda word: word in sentence.lower(), keywords))) ]
    return(relevant)

def get_mentions(keywords, df_to_search, search_numbers = True ):
    keywords = keywords.split(' ')
    if search_numbers == True :
        paper_sentences = [get_single_sentence(keywords,df_to_search.iloc[i]['text'],True) for i in range(len(df_to_search))]
    else:
        paper_sentences = [get_single_sentence(keywords,df_to_search.iloc[i]['text'],False) for i in range(len(df_to_search))]
    df_to_return = pd.DataFrame(data = {'title':df_to_search['title'],'Info':paper_sentences})
    #for i in range(len(df_to_return)):
     #   print(df_to_return.iloc[i])
    return(df_to_return)

def best_database(keywords):
    keywords = keywords.split(' ')
    tf_1 = np.mean([tf_idf_scores1[tf_idf_scores1['token'] == word]['Tf-IDF'].values[0] for word in keywords])
    tf_2 = np.mean([tf_idf_scores2[tf_idf_scores1['token'] == word]['Tf-IDF'].values[0] for word in keywords])
    if tf_1 > tf_2:
        return('new')
    else:
        return('old')


# In[5]:


#Defining the full function:
def knowledge_extraction(abstract_query , detailed_query ,databases ='best', number_of_clusters = 2, search_numbers = True, to_print = False):
    #Parameters:
    #abstract query: The general query to search the tf-idfs vectors to collect the relevant papers
    #detailed query: The detailed query for a specific sentence search on the selected papers 
    #databases : Which one from the two databases should it look for. 
    #Options : Best->The one with the biggest tf-idf total sum, Both -> Return best values on both databases
    #maximum number of clusters which will search for : Higher number -> more complex, more papers, more info
    #search_numbers: Boolean -> wheather or not to look for number on sentence search: eg incubation period wants a number
    #to_print : Boolean -> If it prints all the knowledge extracted
    
    #Checking which is the best database
    if databases =='best':
        paper_collection = best_database(abstract_query)
    
        clusters_to_look = clusters_including(abstract_query, number_of_clusters, paper_collection)
        #number of clusters is concerning the above function
        created_dataframe = get_merged_df(clusters_to_look,paper_collection)
        #merging the dataframes which contain each cluster
        abstracts_include = search_abstracts(abstract_query , created_dataframe)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info = get_mentions(detailed_query,abstracts_include, search_numbers = True )
    else:
        clusters_to_look1 = clusters_including(abstract_query, number_of_clusters, 'new')
        #number of clusters is concerning the above function
        created_dataframe1 = get_merged_df(clusters_to_look1,'new')
        #merging the dataframes which contain each cluster
        abstracts_include1 = search_abstracts(abstract_query , created_dataframe1)
        dataframe_with_info1 = get_mentions(detailed_query,abstracts_include1, search_numbers = True )
        clusters_to_look2 = clusters_including(abstract_query, number_of_clusters, 'old')
        #number of clusters is concerning the above function
        created_dataframe2 = get_merged_df(clusters_to_look2,'old')
        #merging the dataframes which contain each cluster
        abstracts_include2 = search_abstracts(abstract_query , created_dataframe2)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info2 = get_mentions(detailed_query,abstracts_include2, search_numbers = True )
        dataframe_with_info = pd.merge(dataframe_with_info1,dataframe_with_info1,how='outer')
    if to_print==True :
        for i in range(len(dataframe_with_info)):
            for j in range(len(dataframe_with_info.iloc[i]['Info'])):
                print(dataframe_with_info.iloc[i]['Info'][j])
                print(" ")
                print(" ")
            print("____")
            print("Next Paper:",dataframe_with_info.iloc[i]['title'] )
            print("____")
    return(dataframe_with_info)
        


# In[6]:


general = 'incubation'
detailed = 'incubation period'
test = knowledge_extraction(general, detailed, databases ='best', to_print = True)


# ## Gathering values

# In[ ]:


severe_prob = [0.2,0.39,0.44,0.3,0.3,0.1,27.1,0.14,0.15,0.22]
severe_crit = [0.1,0.16,0.05,0.1,0.05,0.05]


# In[47]:


27/53


# In[27]:


#Incubation period
inc_days = [5.9,9.94,5,7,7.45,5.833,5.2,6,7,5.5,7,6.93,5,5.17,6.4,8.13,4.51,15.2,4.6,8.5,14,14,5.4,5.2,5.61,5.14,5,5]
#Interval is the one bellow
inc_range = [(4.78,6.25),(3.93,13.5),(4,10.9),(5.33,6.26),(3.5,9.5),(1,7.8),(1,14),(2.1,8.1),(6,10),(5.6,7.7),(1,14)
             ,(3,11.1),(2,10),(12,16)]


#Infectious period
#Search transmition period
inf_days = [9.9,1.87,1.2,2,10,5,7,15,3,14,2,2,10,3,4.5,9,5,6.25,6.54,5.76,5,]


# ## For comparissons

# In[10]:


def index_finder(word,cluster,collection):
    #Setting a stopping element
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    idx = None
    for j in range(len(tf_idf_df[cluster])):
        if tf_idf_df[cluster].iloc[j]['token'] == word:
            idx = j
    return(idx)
def topic_keywords_lda(words,collection):
    extra = []
    words = words.split(' ')
    if collection == 'new':
        keywords = keywords1
        clusters = clusters1
    else:
        keywords = keywords2
        clusters = clusters2
    #words = sentence.split(' ')
    for cluster in range(clusters):
        for word in words:
            #print(word)
            if word in keywords[cluster]:
                extra.append(cluster)

    return(extra)
 
def clusters_including(query, number_of_clusters, collection):    
    #Returns the tf-idf scores on every cluster.
    #Mean value if more than two words are given
    #number_of_clusters is the number of clusters to return, if not enough docs are found, then i can increase the number
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    words = query.split(' ')
    tf_scores = [np.mean([tf_idf_df[cluster].iloc[index_finder(word,cluster,collection)]['Tf-IDF'] 
                 if index_finder(word,cluster,collection)!= None else 0 for word in words]) for cluster in range(clusters) ]
    #returns two best clusters
    selected_clusters = np.argsort(tf_scores)[::-1][:number_of_clusters]
    #Checking if either word is included on the topics of each cluster extracted by the LDA
    #If so they will also be included
    extra = topic_keywords(words, selected_clusters, collection)
    return(extra)

def get_merged_df(clusters,collection):
#Returns the df which is the merge of the dfs concering each picked cluster
#Input is the output of clusters_including
    if collection == 'new':
        clustered_dfs = clustered_dfs1
        #clusters = clusters1
    else:
        clustered_dfs = clustered_dfs2
        #clusters = clusters2
    picked_dfs = [clustered_dfs[i] for i in clusters]
    merged_df = pd.concat(picked_dfs)
    return(merged_df)
def search_abstracts(sentence, df_to_search):
    #Returns a df where the keywords are included in the abstracts 
    words = sentence.split(' ')
    df_reduced = df_to_search [functools.reduce(lambda a, b: a & b, 
                                                (df_to_search['abstract'].str.contains(word) for word in words))]
    return(df_reduced)
#Here i can define a different search scope, for example incubation, days , range 
#On the pattern search(previous step) i am looking for more abstract searches while here for more detailed ones:
#For example for incubation i can add days as well!!!
#Check the title of the article if it is the same
#Add a formula for pointing if numbers are wanted or not

def get_single_sentence(keywords, text, search_numbers=True):
    #Text is the text of a single paper
    #Splitting text into sentences
    sentences = text.split('. ')
    #search keywords and number (parameter for that)
    if search_numbers == True:
        relevant = [sentence  for sentence in sentences if (all(map(lambda word: word in sentence.lower(), keywords))) 
                    and len(re.findall(r'\d{1,2}',sentence)) ]
    else:
        relevant = [sentence  for sentence in sentences if (all(map(lambda word: word in sentence.lower(), keywords))) ]
    return(relevant)

def get_mentions(keywords, df_to_search, search_numbers = True ):
    keywords = keywords.split(' ')
    if search_numbers == True :
        paper_sentences = [get_single_sentence(keywords,df_to_search.iloc[i]['text'],True) for i in range(len(df_to_search))]
    else:
        paper_sentences = [get_single_sentence(keywords,df_to_search.iloc[i]['text'],False) for i in range(len(df_to_search))]
    df_to_return = pd.DataFrame(data = {'title':df_to_search['title'],'Info':paper_sentences})
    #for i in range(len(df_to_return)):
     #   print(df_to_return.iloc[i])
    return(df_to_return)

def best_database(keywords):
    keywords = keywords.split(' ')
    tf_1 = np.mean([tf_idf_scores1[tf_idf_scores1['token'] == word]['Tf-IDF'].values[0] for word in keywords])
    tf_2 = np.mean([tf_idf_scores2[tf_idf_scores1['token'] == word]['Tf-IDF'].values[0] for word in keywords])
    if tf_1 > tf_2:
        return('new')
    else:
        return('old')


# In[21]:


#Defining the full function:
def knowledge_extraction(abstract_query , detailed_query ,databases ='best', number_of_clusters = 2, search_numbers = True, to_print = False):
    #Parameters:
    #abstract query: The general query to search the tf-idfs vectors to collect the relevant papers
    #detailed query: The detailed query for a specific sentence search on the selected papers 
    #databases : Which one from the two databases should it look for. 
    #Options : Best->The one with the biggest tf-idf total sum, Both -> Return best values on both databases
    #maximum number of clusters which will search for : Higher number -> more complex, more papers, more info
    #search_numbers: Boolean -> wheather or not to look for number on sentence search: eg incubation period wants a number
    #to_print : Boolean -> If it prints all the knowledge extracted
    
    #Checking which is the best database
    if databases =='best':
        paper_collection = best_database(abstract_query)
    
        clusters_to_look = clusters_including(abstract_query, number_of_clusters, paper_collection)
        #number of clusters is concerning the above function
        created_dataframe = get_merged_df(clusters_to_look,paper_collection)
        #merging the dataframes which contain each cluster
        abstracts_include = search_abstracts(abstract_query , created_dataframe)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info = get_mentions(detailed_query,abstracts_include, search_numbers = True )
    else:
        clusters_to_look1 = clusters_including(abstract_query, number_of_clusters, 'new')
        #number of clusters is concerning the above function
        created_dataframe1 = get_merged_df(clusters_to_look1,'new')
        #merging the dataframes which contain each cluster
        abstracts_include1 = search_abstracts(abstract_query , created_dataframe1)
        dataframe_with_info1 = get_mentions(detailed_query,abstracts_include1, search_numbers = True )
        clusters_to_look2 = clusters_including(abstract_query, number_of_clusters, 'old')
        #number of clusters is concerning the above function
        created_dataframe2 = get_merged_df(clusters_to_look2,'old')
        #merging the dataframes which contain each cluster
        abstracts_include2 = search_abstracts(abstract_query , created_dataframe2)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info2 = get_mentions(detailed_query,abstracts_include2, search_numbers = True )
        dataframe_with_info = pd.merge(dataframe_with_info1,dataframe_with_info1,how='outer')
    if to_print==True :
        for i in range(len(dataframe_with_info)):
            for j in range(len(dataframe_with_info.iloc[i]['Info'])):
                print(dataframe_with_info.iloc[i]['Info'][j])
                print(" ")
                print(" ")
            print("____")
            print("Next Paper:",dataframe_with_info.iloc[i]['title'] )
            print("____")
    return(dataframe_with_info)
def check(z):
    if len(z)==0:
        return(0)
    else: return(1)       


# In[15]:


#LDA
def knowledge_extraction_lda(abstract_query , detailed_query ,databases ='both', 
                             number_of_clusters = 2, search_numbers = False, to_print = False):
#Searching on both databases to increase number of results
    clusters_to_look1 = topic_keywords_lda(abstract_query,'new')
    if len(clusters_to_look1) != 0:
        #number of clusters is concerning the above function
        created_dataframe1 = get_merged_df(clusters_to_look1,'new')
            #merging the dataframes which contain each cluster
        abstracts_include1 = search_abstracts(abstract_query , created_dataframe1)
        dataframe_with_info1 = get_mentions(detailed_query,abstracts_include1, search_numbers = True )
        #return(dataframe_with_info1)

    #clusters_to_look2 = topic_keywords_lda(abstract_query,'old')
    clusters_to_look2 = []
    no_res = False
    #if len(clusters_to_look2) != 0:
        #number of clusters is concerning the above function
    #    created_dataframe2 = get_merged_df(clusters_to_look2,'old')
            #merging the dataframes which contain each cluster
    #    abstracts_include2 = search_abstracts(abstract_query , created_dataframe2)
            #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
   #     dataframe_with_info2 = get_mentions(detailed_query,abstracts_include2, search_numbers = search_numbers )
    if len(clusters_to_look2) == 0 and len(clusters_to_look1) == 0:
        print('no results')
        no_res = True
    elif len(clusters_to_look2) != 0 and len(clusters_to_look1) != 0:
        
        dataframe_with_info = pd.concat([dataframe_with_info1 , dataframe_with_info2] )
    elif len(clusters_to_look2) ==0 and len(clusters_to_look1) !=0:
        dataframe_with_info  = dataframe_with_info1
    else:
        dataframe_with_info  = dataframe_with_info2
    if no_res ==False:
        dataframe_with_info['tokeep'] = dataframe_with_info.apply(lambda row : check(row['Info']),axis=1)
        dataframe_with_info = dataframe_with_info[dataframe_with_info['tokeep']==1]
        del dataframe_with_info['tokeep']
        if to_print==True :
            for i in range(len(dataframe_with_info)):
                if len(dataframe_with_info.iloc[i]['Info']) != 0:
                    for j in range(len(dataframe_with_info.iloc[i]['Info'])):
                        print(dataframe_with_info.iloc[i]['Info'][j])
                        print(" ")
                        print(" ")
                    print("____")
                    print("Next Paper:",dataframe_with_info.iloc[i]['title'] )
                    print("____")
                    #print(len(clusters_to_look2))
        return(dataframe_with_info)
    else:
        print("No Res")


# In[159]:


#Incubation period

#Period, incubation , exposed=>Nothing
#incubation = 0 + 0 
#period + incubation period = 0 +11 
#days + incubation days >70 + 71
#days + incubation period -> 79 + 81
#symptom + incubation period -> 88 + 88
#Symptom + incubation days -> 65 + 61

general = 'symptom'
detailed = 'incubation days'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[86]:


#220 results after 4 rephrases


# In[65]:


#Infectious

#infectious return 0
#infections = 0 + 0
#period + infectious = 5 + 30
#day + infectious day = 47 + 44
#day + infectious duration = 15 + 10
#symptom + infetious period = 46 + 41
#symptom + infectious duration =13 + 28



#Infectios nothing, 
general = 'symptom'
detailed = 'infectious days'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[60]:


keywords1


# In[73]:


#Assymptomatics

#asymptomatics + out = 0 + 61
#symptom returned 0
#perchentage returned 0
#probability +assymptomaics return 0 + 17
#probability + mild return 0 + 6
#probability + no symptoms return 22 + 15

#Infectios nothing, 
general = 'probability'
detailed = 'no symptoms'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[79]:


#critical

#critical returned 0
#symptom returned 0

#severe + critical symptoms 138 + 30
#severe + critical cases 340 + 73

#Infectios nothing, 
general = 'severe'
detailed = 'critical cases'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[84]:


#severe


 
#probability + severe = 32 + 18
#severe + severe probability = 28 +5 
#Severe + severe symptoms out(out for out of ... people, ... had severe symptoms) 139 + 37
#severe + severe proportion = 205 + 53
general = 'severe'
detailed = 'severe proportion'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[91]:


#Hospital days

#admission + days = 51
#admission + duration = 27
#hospital +hospitalization duration = 42 + 44
#hospital + hospitalization days =142 + 144
#hospital + hospitalization period = 47 + 53
general = 'admission'
detailed = 'hospitalization days'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[97]:


#Critical days

 
#ICU = 0
#critical = 0
#severe + ICU = 0
#admission + icu duration = 12 
#hospital + icu duration = 49 + 51
#hospital + icu days = 144 + 147 
#hospital + ventilation days = 70 + 69
general = 'admission'
detailed = 'icu duration'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# In[99]:


#mortality 

#mortality + mortality rate = 418 + 453

general = 'mortality'
detailed = 'mortality rate'
zz = knowledge_extraction_lda(general,detailed,to_print=False)
len(zz)


# ## TF - IDF only

# In[103]:


def clusters_including_tf(query, number_of_clusters, collection):    
    #Returns the tf-idf scores on every cluster.
    #Mean value if more than two words are given
    #number_of_clusters is the number of clusters to return, if not enough docs are found, then i can increase the number
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    words = query.split(' ')
    tf_scores = [np.mean([tf_idf_df[cluster].iloc[index_finder(word,cluster,collection)]['Tf-IDF'] 
                 if index_finder(word,cluster,collection)!= None else 0 for word in words]) for cluster in range(clusters) ]
    #returns two best clusters
    selected_clusters = np.argsort(tf_scores)[::-1][:number_of_clusters]

    return(selected_clusters)

def check(z):
    if len(z)==0:
        return(0)
    else: return(1)  
def knowledge_extraction_tf(abstract_query , detailed_query ,databases ='best', number_of_clusters = 2, search_numbers = False, to_print = False):
    #Parameters:
    #abstract query: The general query to search the tf-idfs vectors to collect the relevant papers
    #detailed query: The detailed query for a specific sentence search on the selected papers 
    #databases : Which one from the two databases should it look for. 
    #Options : Best->The one with the biggest tf-idf total sum, Both -> Return best values on both databases
    #maximum number of clusters which will search for : Higher number -> more complex, more papers, more info
    #search_numbers: Boolean -> wheather or not to look for number on sentence search: eg incubation period wants a number
    #to_print : Boolean -> If it prints all the knowledge extracted
    
    #Checking which is the best database
    if databases =='best':
        #skipping looking for the best database, will only look at the newest
        #paper_collection = best_database(abstract_query)
        paper_collection = 'new'
        #In clusers_to_look for tf_idf only I am chaning the clusters_including function to not return lda
        clusters_to_look = clusters_including_tf(abstract_query, number_of_clusters, paper_collection)
        #number of clusters is concerning the above function
        created_dataframe = get_merged_df(clusters_to_look,paper_collection)
        #merging the dataframes which contain each cluster
        abstracts_include = search_abstracts(abstract_query , created_dataframe)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info = get_mentions(detailed_query,abstracts_include, search_numbers = True )
    else:
        clusters_to_look1 = clusters_including(abstract_query, number_of_clusters, 'new')
        #number of clusters is concerning the above function
        created_dataframe1 = get_merged_df(clusters_to_look1,'new')
        #merging the dataframes which contain each cluster
        abstracts_include1 = search_abstracts(abstract_query , created_dataframe1)
        dataframe_with_info1 = get_mentions(detailed_query,abstracts_include1, search_numbers = True )
        clusters_to_look2 = clusters_including(abstract_query, number_of_clusters, 'old')
        #number of clusters is concerning the above function
        created_dataframe2 = get_merged_df(clusters_to_look2,'old')
        #merging the dataframes which contain each cluster
        abstracts_include2 = search_abstracts(abstract_query , created_dataframe2)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
        dataframe_with_info2 = get_mentions(detailed_query,abstracts_include2, search_numbers = True )
        dataframe_with_info = pd.merge(dataframe_with_info1,dataframe_with_info1,how='outer')
    dataframe_with_info['tokeep'] = dataframe_with_info.apply(lambda row : check(row['Info']),axis=1)
    dataframe_with_info = dataframe_with_info[dataframe_with_info['tokeep']==1]
    del dataframe_with_info['tokeep']
    if to_print==True :
        for i in range(len(dataframe_with_info)):
            for j in range(len(dataframe_with_info.iloc[i]['Info'])):
                print(dataframe_with_info.iloc[i]['Info'][j])
                print(" ")
                print(" ")
            print("____")
            print("Next Paper:",dataframe_with_info.iloc[i]['title'] )
            print("____")
    return(dataframe_with_info)
     


# In[160]:


#Incubation period

#Period, incubation , exposed=>Nothing
#period + incubation days = 62
#Incubation + incubation days -> 9 + 36
#Incubation + incubation period -> 16 + 44
#days + incubation days -> 47 + 34
#days + incubation period -> 16 + 39
#symptom + incubation period -> 88 + 53
#Symptom + incubation days -> 65 + 37
general = 'period'
detailed = 'incubation days'
zz = knowledge_extraction_tf(general,detailed,to_print=False)
len(zz)


# In[117]:


#Infectious


#infectious + infectious days = 15 + 61
#infectious + infectious period = 16 + 70
#infectious + infectious duration = 7 + 19
#day + infectious day = 32 + 22
#day + infectious duration = 9 + 1
#symptom + infetious period = 32 + 14
#symptom + infectious duration =13 + 2



general = 'symptom'
detailed = 'infectious duration'
zz = knowledge_extraction_tf(general,detailed,to_print=False)
len(zz)


# In[126]:


#Assymptomatics

#asymptomatics + out = 0 + 61
#symptom returned 0
#perchentage returned 0
#probability +assymptomaics return 0 + 17
#probability + mild return 0 + 6
#probability + no symptoms return 22 + 15


general = 'probability'
detailed = 'no symptoms' 
zz = knowledge_extraction_tf(general,detailed,to_print=False)
len(zz)


# In[13]:


#critical

#critical + critical cases
#critical + critical out =
#severe + critical symptoms = 
#severe + critical cases = 

#Infectios +critical cases
#infections + critical symptoms

general = 'infection'
detailed = 'critical cases'
zz = knowledge_extraction_tf(general,detailed,to_print=False)
len(zz)


# In[134]:


#severe

#probability + severe = 55 + 39
#severe + severe probability = 12 + 4
#Severe + severe symptoms out(out for out of ... people, ... had severe symptoms) 57 + 29
#severe + severe proportion = 75 + 39
general = 'severe'
detailed = 'severe proportion'
zz = knowledge_extraction_tf(general,detailed,to_print= False)
len(zz)


# In[138]:


#Hospital days
 
#hospital +hospitalization duration = 47 + 32
#hospital + hospitalization days =142 + 110
#hospital + hospitalization period = 48 + 37
#admission + hospitalization duration
#admission + hospitalization days
#admission + hospitalization period
general = 'hospital'
detailed = 'hospitalization period'
zz = knowledge_extraction_tf(general,detailed,to_print=False)
len(zz)


# In[143]:


#Critical days

 
#ICU + ICU days= 0
#ICU + ICU duration= 0
#critical + critical days = 27 + 37
#severe + severe icu = 263 + 169
#hospital + icu duration = 43 +41
#hospital + icu days = 140 + 155
#hospital + ventilation days = 94 + 73
general = 'critical'
detailed = 'critical day'
zz = knowledge_extraction_tf(general,detailed,to_print= False)
len(zz)


# In[144]:


#mortality 

#mortality + mortality rate = 375 + 374

general = 'mortality'
detailed = 'mortality rate'
zz = knowledge_extraction_tf(general,detailed,to_print=True)
len(zz)


# ## Tf-IDF + LDA

# In[233]:


asd = [1,1,2,2,3,5]

list(set(asd)) 


# In[8]:


def topic_keywords_both(words,current_selected,collection):
    extra = []
    #words = words.split(' ')
    if collection == 'new':
        keywords = keywords1
        clusters = clusters1
    else:
        keywords = keywords2
        clusters = clusters2
    #words = sentence.split(' ')
    for cluster in range(clusters):
        for word in words:
            #print(word)
            if word in keywords[cluster]:
                extra.append(cluster)
    for new_cluster in extra:
        if new_cluster not in current_selected:
            #print(new_cluster)
            current_selected = np.append(current_selected,new_cluster)
    return(current_selected)
 



def clusters_including_both(query, number_of_clusters, collection):    
    #Returns the tf-idf scores on every cluster.
    #Mean value if more than two words are given
    #number_of_clusters is the number of clusters to return, if not enough docs are found, then i can increase the number
    if collection == 'new':
        tf_idf_df = tf_idf_df1
        clusters = clusters1
    else:
        tf_idf_df = tf_idf_df2
        clusters = clusters2
    words = query.split(' ')
    tf_scores = [np.mean([tf_idf_df[cluster].iloc[index_finder(word,cluster,collection)]['Tf-IDF'] 
                 if index_finder(word,cluster,collection)!= None else 0 for word in words]) for cluster in range(clusters) ]
    #returns two best clusters
    selected_clusters = np.argsort(tf_scores)[::-1][:number_of_clusters]
    #Checking if either word is included on the topics of each cluster extracted by the LDA
    #If so they will also be included
    extra = topic_keywords_both(words, selected_clusters,collection)

    #print(extra)
    return(extra)

def check(z):
    if len(z)==0:
        return(0)
    else: return(1)  
def knowledge_extraction_both(abstract_query , detailed_query ,databases ='best', number_of_clusters = 2, search_numbers = False, to_print = False):
    #Parameters:
    #abstract query: The general query to search the tf-idfs vectors to collect the relevant papers
    #detailed query: The detailed query for a specific sentence search on the selected papers 
    #databases : Which one from the two databases should it look for. 
    #Options : Best->The one with the biggest tf-idf total sum, Both -> Return best values on both databases
    #maximum number of clusters which will search for : Higher number -> more complex, more papers, more info
    #search_numbers: Boolean -> wheather or not to look for number on sentence search: eg incubation period wants a number
    #to_print : Boolean -> If it prints all the knowledge extracted
    
    #Checking which is the best database

        #skipping looking for the best database, will only look at the newest
        #paper_collection = best_database(abstract_query)
    paper_collection = 'new'
        #In clusers_to_look for tf_idf only I am chaning the clusters_including function to not return lda
    clusters_to_look = clusters_including_both(abstract_query, number_of_clusters, paper_collection)
        #number of clusters is concerning the above function
    created_dataframe = get_merged_df(clusters_to_look,paper_collection)
        #merging the dataframes which contain each cluster
    abstracts_include = search_abstracts(abstract_query , created_dataframe)
        #Keeping only the papers on the merged dataframe which have the query mentioned on the abstract
    dataframe_with_info = get_mentions(detailed_query,abstracts_include, search_numbers = True )
    #return(dataframe_with_info)
    dataframe_with_info['tokeep'] = dataframe_with_info.apply(lambda row : check(row['Info']),axis=1)
    dataframe_with_info = dataframe_with_info[dataframe_with_info['tokeep']==1]
    del dataframe_with_info['tokeep']
    if to_print==True :
        for i in range(len(dataframe_with_info)):
            for j in range(len(dataframe_with_info.iloc[i]['Info'])):
                print(dataframe_with_info.iloc[i]['Info'][j])
                print(" ")
                print(" ")
            print("____")
            print("Next Paper:",dataframe_with_info.iloc[i]['title'] )
            print("____")
    return(dataframe_with_info)
     


# In[20]:


#Incubation period

#Period + incubation days = 1 + 67??
#Incubation + incubation days -> 35 + 36
#Incubation + incubation period -> 43 + 44
#days + incubation days -> 55 + 71
#days + incubation period -> 56 +81
#symptom + incubation period -> 100 + 88
#Symptom + incubation days -> 74 + 61


general = 'incubation'
detailed = 'incubation period days'
zz = knowledge_extraction_both(general,detailed,to_print=True)


# In[18]:


#Infectious

#infectious + infectious days = 59 + 61
#infectious + infectious period = 73 + 70
#infectious + infectious duration = 20 + 19
#day + infectious day = 56 + 38
#day + infectious duration = 20 + 10
#symptom + infetious period = 41 + 8
#symptom + infectious duration =14 + 8

general = 'symptom'
detailed = 'infectious period days'
zz = knowledge_extraction_both(general,detailed,to_print=True)
len(zz)


# In[16]:


zz


# In[190]:


#Assymptomatics

#assymptomatics + assymptomatics proportion = 27 +0 
#symptom + mild out = 149 + 134
#symptom + asymptomatic = 533 + 479
#perchentage returned 0
#probability +assymptomaics = 23 + 34
#probability + mild = 16 + 21
#probability + no symptoms = 20 + 23


general = 'probability'
detailed = 'no symptoms'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[194]:


#severe

#probability + severe = 55 + 39
#severe + severe probability = 30 + 8
#Severe + severe symptoms out(out for out of ... people, ... had severe symptoms) 73 + 47
#severe + severe proportion = 110 + 63


general = 'probability'
detailed = 'severe'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[198]:


#critical

#critical + critical cases = 86 + 86
#critical + critical out = 108 + 90
#severe + critical symptoms = 102 + 56
#severe + critical cases = 223 + 92

#critical + critical cases
#critical + critical out =
#severe + critical symptoms = 
#severe + critical cases = 

#Infectios +critical cases
#infections + critical symptoms

general = 'severe'
detailed = 'critical cases'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[201]:


#Hospital days
 
#hospital +hospitalization duration = 41 + 44
#hospital + hospitalization days =142 + 144
#hospital + hospitalization period = 47 + 53
#admission + hospitalization duration
#admission + hospitalization days
#admission + hospitalization period
general = 'hospital'
detailed = 'hospitalization duration'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[208]:


#Critical days

#ICU + ICU days= 111 + 102
#ICU + ICU duration= 37 + 33
#critical + critical days = 24 + 25
#severe + severe icu = 386 + 229
#hospital + icu duration = 49 + 51
#hospital + icu days = 144 + 147
#hospital + ventilation days = 70 + 69

general = 'icu'
detailed = 'icu duration'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[12]:


general = 'icu'
detailed = 'icu days'
zz = knowledge_extraction_both(general,detailed,to_print=True)
len(zz)


# In[209]:


#mortality 

#mortality + mortality rate = 521 + 453
general = 'mortality'
detailed = 'mortality rate'
zz = knowledge_extraction_both(general,detailed,to_print=False)
len(zz)


# In[10]:


general = 'mortality'
detailed = 'mortality rate'
zz = knowledge_extraction_both(general,detailed,to_print=True)
len(zz)


# In[ ]:





# In[ ]:


#Tonigh, repeat it all for 2 more times, keep results by adding them next to each other and take mean
#Talk results say what I have said on the notes, in general both produce results, tf-idf more steadily,
#lda needs to find the perfect match. Combination optimal

#Morning take values from each, create comparisson, say no big difference does exist(comparison with true values)
#conclude that the combination returns more results and works best as it hides limitations of both methods
#Final performance of calibrated method and the mean of the features comparison(take calibrated as true)


# In[ ]:


#Visualization: total reprheases on x-axis(For all features, start with 1 go to total rephrases)
#say some required more and some less(for example mortality and assymptomatics)
#In general state that lda performed when hit match(big increase when find keyword) best but needed more reprhases, examples, 
#while tf steady increase on results

#Both methods work, lda needs more reprhases

