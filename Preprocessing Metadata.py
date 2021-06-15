#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm.auto import tqdm
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

import spacy
from spacy_langdetect import LanguageDetector
import en_core_sci_lg
import os
import umap


# ### Getting the Metadata csv.
# 
# 
# * Keeping the columns needed

# In[2]:


#First dataset contains the abstracts and the publish time.
#Will focus on these!
metadata = pd.read_csv('D:/complex data project/metadata.csv',
                       usecols=['sha','title','abstract','authors','publish_time'])


# Checking the length

# In[3]:


len(metadata)


# **Observation**
# 
# Several articles are from different times talking about other similar viruses or concepts relevant to COVID-19
# 
# The novel coronavirus first appeared in late 2019 and early 2020
# 
# Naturally, the first papers were written 

# In[4]:


#Converting time to datetime
metadata['publish_time'] = pd.to_datetime(metadata['publish_time'])
#Extracting just the year from each datetime
years = [metadata.iloc[i]['publish_time'].year for i in range(len(metadata)) if metadata.iloc[i]['publish_time'].year < 2020 ]
print(len(years))

#Plotting
plt.figure(figsize = (16,8))
plt.hist(years)
plt.show()


# In[5]:


#Keeping only articles from 2020 as I assume that only these articles contain valuable information
#There might some articles from past times(like 2019) which give relevant information but in this work there are not included
def new_articles(date):
    if date.year == 2020:
        return(True)
    else:
        return(False)
    
metadata['to_keep'] = metadata.apply(lambda row: new_articles(row['publish_time']),axis = 1 )
metadata = metadata[metadata['to_keep']==True]
del metadata["to_keep"]


# In[6]:


#I have reduced the number to just bellow 190.000
len(metadata)


# In[49]:


#What is more, several articles talk about pneumonia, sars, or a similar viruses.
#These articles are not of my concern
#My Focus is intererely on Covid
#Again I assume that Covid-relevant info is only on articles containing one of the specific keywords on their abstract(or title)
#Rarely an article wont include such a term on their abstract(or title)

#Keeping only the articles which contain Covid-19 or sth similar to abstract or title

#These are the 4 most widely available notations of covid-19

def get_relevant(df):
    cleaned = df[(df['check_for_covid'].str.contains('covid'))|
                (df['check_for_covid'].str.contains('-cov-2'))|
                (df['check_for_covid'].str.contains('cov2'))|
                (df['check_for_covid'].str.contains('ncov'))]
    return(cleaned)

metadata = metadata.fillna('empty')
#adding the title to the abstract so i wont check both the abstract and the title
#lowercasing as well
metadata["check_for_covid"] = metadata["abstract"].str.lower() + metadata["title"].str.lower()
metadata = get_relevant(metadata)
metadata = metadata.drop_duplicates(subset='title', keep="first")
del metadata["check_for_covid"]


# In[50]:


#As most of my approach will be focused on abstracts
#I am removing the texts with no abstracts

metadata = metadata[metadata['abstract'] != 'empty']
len(metadata)


# In[51]:


#Next I am going to focus on the language of the papers
#As my set of techniques focuses on English language
#I will only focus on papers containing english words.

#To detect the language I will use spacy which is for language processing of scientific articles
#Ref: https://spacy.io/universe/project/spacy-langdetect

#Will pick a specific model designed for mining scientific papers
#The model was selected from:
#https://allenai.github.io/scispacy/
#https://github.com/allenai/scispacy


# In[52]:


#Initializing the pipeline of the used model
nlp = en_core_sci_lg.load()

#Adding a Language Detector
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


# In[53]:


#A Small demonstration of the usage of the language detector added on the model's pipeline:
test = nlp('Hey Everyone')
test1 = nlp("Mi nombre es Filotas")
if (test1._.language['language'] == "en"):
    print(True, 'It is English')
else:
    print(test1._.language['language'])


# In[54]:


if (test._.language['language'] == "en"):
    print(True)
else:
    print(test._.language['language'])


# In[55]:


def check_language(abstract):
    abstract_nlp = nlp(abstract)
    language = abstract_nlp._.language['language']
    return(language)

#Applying the model on the Abstracts was extremely time-consuming
#As a result and after checking the perfromance of the language detector
#Apllying it on the abstracts
metadata['Language'] = metadata.apply(lambda row: check_language(str(row['title'])) , axis = 1)


# In[56]:


#Checking Languages:
languages = metadata.groupby('Language').agg('count').reset_index()[['Language','title']].sort_values('title')
plt.figure(figsize = (12,8))
plt.barh(languages['Language'],languages['title'])
plt.show()


# In[57]:


#English is the most dominant language with some spanish and french articles
#Keeping only the English Articles

metadata = metadata[metadata['Language'] == 'en']
del metadata['Language']
len(metadata)


# In[58]:


#Saving so I wont have to do the same steps again

#import os
#filepath = 'D:/complex data project'
#name = os.path.join(filepath , 'metadata1.csv')
#metadata.to_csv(name)


# In[59]:


#metadata = pd.read_csv('D:/complex data project/metadata1.csv')


# In[60]:


#Dropping some duplicates:
metadata = metadata.drop_duplicates()
len(metadata)


# In[62]:


#A total of 62.330 Articles!

#A very big dataset concerning the length of each paper

#My approach to reduce the volume even more is target months
#Rearch regarding Covid is assumed to be into two parts!

#First part regards the outbreak of the virus in Western World(Italy)
#Uncertainty was extremely high till that point and researchers focused on identyfing the mechanisms of the virus
#This period was until lockdown guidelines were lifted at about May

#Then the next period is after May where researchers focused on tackling and stopping the spread of the virus

#These assumptions might be uneccessary, but to tackle memory issues raised by my computer(Even Colab or Spark FAILED)
#I will split the knowledge extractor into two periods.
#Till May and after May


# In[61]:


check_dates = [metadata.iloc[i]['publish_time'].month for i in range(len(metadata))]
plt.figure(figsize=(16,8))
plt.hist(check_dates)
plt.show()


# In[71]:


metadata['month_of_publish'] = check_dates
very_early_papers = metadata[metadata['month_of_publish'] <= 2]
mid_period_papers = metadata[(metadata['month_of_publish'] > 2) & (metadata['month_of_publish'] <=5 ) ]
late_papers = metadata[metadata['month_of_publish'] > 5]
print("Total of Early Papers:" , len(very_early_papers))
print("Total of Medium-Period Papers:" , len(mid_period_papers))
print("Total of Late Papers:" , len(late_papers))


# In[72]:


#Validating with my primary work
#metadata1 = pd.read_csv('D:/complex data project/metadata1.csv')
#metadata1 = metadata1[metadata1['abstract']!='empty']
#metadata1 = metadata1[metadata1['to_keep']==True].drop_duplicates()
#metadata1


# In[74]:


#Some papers had their title as abstract so I am removing them as well!
metadata['title'] = metadata['title'].str.lower()
metadata['abstract'] = metadata['abstract'].str.lower()
metadata = metadata[metadata['title']!=metadata['abstract']]


# In[77]:


filepath = 'D:/complex data project'
name = os.path.join(filepath , 'step1.csv')
metadata.to_csv(name)

