#This files extracts knowledge from the document collection.
# Users makes some queries regarding some properties of the virus and the functions searches the collection for answers
# It returns the sentences with the exact properties

# Approach:
# The user makes two queries. 
# A general and a specific one. The general regards the abstract topic the user is interested in. Then on the detailed he/she gets the exact information
#Example:
# General query: mortality
# Detailed: mortality rate
# General: hospital
# Detailed: icu duration

# There are two collection of documents (before and after May)
# The default collection is the one after May
# User can select to search either collections
# In case not enough information is generated from on collection user can select to search both collections

# METHDOLOGY
# For each cluster on both collection the aggregated TF-iDF of every keyword is calculated.
# For every general query, the two clusters with the higher aggregated TF-iDF (on the specific keywords) are selected.
# In addition, the topic keywords extracted from LDA for every cluster are also explored.
# If a query matches a topic keyword, then the cluster with the topic keyword is also selected

#Then on the selected clusters
#Sentences which include the detailed query are returned

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



df1 = pd.read_csv('D://Uni_Stuff/complex data project/df_final111.csv')
df2 = pd.read_csv('D:/Uni_Stuff/complex data project/df_final22.csv')
pkl_file = open('D:/Uni_Stuff/COVID-19/keywords111.pkl', 'rb')
keywords1 = pickle.load(pkl_file)
pkl_file = open('D:/Uni_Stuff/COVID-19/keywords22.pkl', 'rb')
keywords2 = pickle.load(pkl_file)


pkl_file = open('D:/Uni_Stuff/COVID-19/keywords111.pkl', 'rb')
keywords1 = pickle.load(pkl_file)

pkl_file = open('D:/Uni_Stuff/COVID-19/keywords1.pkl', 'rb')
keywords11 = pickle.load(pkl_file)


#Aggregated TF-iDF 

#Create a tf-idf vectorizer for every cluster and add the sums
#First
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


#Will get the tf-idf score of both databases and from every cluster within them both
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


#Creating a vectorizer for every dataset collection
#Will get the tf-idf score again as it was time-consuming loading it
vocabulary = df1['text']
#Fitting
vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized1 = vectorizer.fit_transform(tqdm(vocabulary))

#Moving to the second paper collection
vocabulary = df2['text']
#Fitting
vectorizer1 = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 20) #Might change min_df to 1 if no results are picked
tqdm.pandas()
text_vectorized2 = vectorizer1.fit_transform(tqdm(vocabulary))

#Getting the tf-idf scores for every term on the two collections
vectorized_df1 = pd.DataFrame(columns = vectorizer.get_feature_names() , data = text_vectorized1.toarray())
sums = np.array([np.sum(vectorized_df1[j]) for j in vectorized_df1.columns])
tf_idf_scores1 = pd.DataFrame(index = vectorized_df1.columns , data = sums).rename(columns = {0:'Tf-IDF'})
tf_idf_scores1 = tf_idf_scores1.sort_values('Tf-IDF' , ascending = False)
tf_idf_scores1 = tf_idf_scores1.reset_index().rename(columns = {'index':'token'})

#Getting the tf-idf scores for every term on the two collections
vectorized_df2 = pd.DataFrame(columns = vectorizer1.get_feature_names() , data = text_vectorized2.toarray())
sums = np.array([np.sum(vectorized_df2[j]) for j in vectorized_df2.columns])
tf_idf_scores2 = pd.DataFrame(index = vectorized_df2.columns , data = sums).rename(columns = {0:'Tf-IDF'})
tf_idf_scores2 = tf_idf_scores2.sort_values('Tf-IDF' , ascending = False)
tf_idf_scores2 = tf_idf_scores2.reset_index().rename(columns = {'index':'token'})




#Creating the vectorizers
vectorizers1 = []
#vectorizers2 = []
for i in range(clusters1): 
    vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer , min_df = 5, max_df = 0.9)
    vectorizers1.append(vectorizer)
   

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


#Defining the functions for the knowledge extractor


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
 
#get the clusters 
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
#Defining the full function:
def knowledge_extraction(abstract_query , detailed_query ,databases ='best', number_of_clusters = 2, search_numbers = False, to_print = False):
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


#Examples for values regarding the incubatio period
general = 'incubation'
detailed = 'incubation period'
test = knowledge_extraction(general, detailed, databases ='best', to_print = True)
general = 'symptom'
detailed = 'incubation days'
test = knowledge_extraction(general,detailed,to_print=False)


#Ouputs collected:
#Incubation period
inc_days = [5.9,9.94,5,7,7.45,5.833,5.2,6,7,5.5,7,6.93,5,5.17,6.4,8.13,4.51,15.2,4.6,8.5,14,14,5.4,5.2,5.61,5.14,5,5]
#Interval are also returned below
inc_range = [(4.78,6.25),(3.93,13.5),(4,10.9),(5.33,6.26),(3.5,9.5),(1,7.8),(1,14),(2.1,8.1),(6,10),(5.6,7.7),(1,14)
             ,(3,11.1),(2,10),(12,16)]



general = 'symptom'
detailed = 'incubation days'
test = knowledge_extraction(general,detailed,to_print=False)

#Infectious period:
general = 'symptom'
detailed = 'infectious days'
test = knowledge_extraction(general,detailed,to_print=False)
#Search transmition period
general = 'symptom'
detailed = 'infectious duration'
test = knowledge_extraction(general,detailed,to_print=False)

inf_days = [9.9,1.87,1.2,2,10,5,7,15,3,14,2,2,10,3,4.5,9,5,6.25,6.54,5.76,5,]

#Assymptomatics
general = 'probability'
detailed = 'no symptoms'
test = knowledge_extraction(general,detailed,to_print=False)

#critical condition 
general = 'severe'
detailed = 'critical cases'
test = knowledge_extraction(general,detailed,to_print=False)
#severe probability 
general = 'severe'
detailed = 'severe proportion'
test = knowledge_extraction(general,detailed,to_print=False)
general = 'probability'
detailed = 'severe'
test = knowledge_extraction(general,detailed,to_print=False)
severe_prob = [0.2,0.39,0.44,0.3,0.3,0.1,27.1,0.14,0.15,0.22]
severe_crit = [0.1,0.16,0.05,0.1,0.05,0.05]

#Hospital days
general = 'admission'
detailed = 'hospitalization days'
test = knowledge_extraction(general,detailed,to_print=False)
general = 'hospital'
detailed = 'hospitalization period'
test = knowledge_extraction(general,detailed,to_print=False)

#Critical days
general = 'admission'
detailed = 'icu duration'
test = knowledge_extraction(general,detailed,to_print=False)
general = 'critical'
detailed = 'critical day'
test = knowledge_extraction(general,detailed,to_print= False)
general = 'icu'
detailed = 'icu duration'
test = knowledge_extraction(general,detailed,to_print=False)
general = 'icu'
detailed = 'icu days'
test = knowledge_extraction(general,detailed,to_print=True)

#mortality 
general = 'mortality'
detailed = 'mortality rate'
test = knowledge_extraction(general,detailed,to_print=False)
