#Initial preprocessing techniques are described here.
#The main objective is to discard papers found in literature which do not include valuable infromation
#The total length of the dataset will be reduced. 
#This results in computationaly favourable operations

#The steps followed in this file include:
# (1) Keep only papers talking about Covid-19
# (2) Discard old (before 2019) articles
# (3) Remove papers with missing infromation (papers with no abstract)
# (4) Remove non-english papers
# (5) Make a strong assumption to further reduce the total number of papers
#     The aim is to make the final dataset computationaly affordable.

import warnings
warnings.filterwarnings('ignore')

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


# Getting the Metadata csv (This includes information regarding the papers found in literature)
# Keeping the columns needed

metadata = pd.read_csv('D:/complex data project/metadata.csv',
                       usecols=['sha','title','abstract','authors','publish_time'])


# Checking the length
print(len(metadata))
# Output: 
# 62330 

#Considering that 62330 full-text papers will crash the system the aim is to reduce this number

# **Observation**
# Several articles come from different times talking about other similar viruses or concepts relevant to COVID-19
# The novel coronavirus first appeared in late 2019s and early 2020s
# Naturally, the first papers were written in that period

#Converting time to datetime
metadata['publish_time'] = pd.to_datetime(metadata['publish_time'])
#Extracting just the year from each datetime
years = [metadata.iloc[i]['publish_time'].year for i in range(len(metadata)) if metadata.iloc[i]['publish_time'].year < 2020 ]

#Running a histogram to analyze the distribution of the papers throughout the years

#Plotting
plt.figure(figsize = (16,8))
plt.hist(years)
plt.show()

#Keeping only articles from 2020.
#I assume that only these articles contain valuable information
#This is not a very strong assumption.
#There might some articles from past times(like 2019) which give relevant information but in this work there are not included
def new_articles(date):
    if date.year == 2020:
        return(True)
    else:
        return(False)
    
metadata['to_keep'] = metadata.apply(lambda row: new_articles(row['publish_time']),axis = 1 )
metadata = metadata[metadata['to_keep']==True]
del metadata["to_keep"]

#I have reduced the number to just bellow 190.000
len(metadata)
#Output 189304

#Several articles talk about pneumonia, sars, or a similar viruses.
#I assume that these articles do not include valuable information
#My Focus is intererely on Covid-19
#My hypothesis is that Covid-19 relevant information is included only on articles containing a specific keyword on their abstract(or title)
#Rarely an article wont include such a term on their abstract(or title)

#Keeping only the articles which contain Covid-19 or a similar keyword or their abstract or title
#These are the 4 most widely available notations of covid-19

def get_relevant(df):
    cleaned = df[(df['check_for_covid'].str.contains('covid'))|
                (df['check_for_covid'].str.contains('-cov-2'))|
                (df['check_for_covid'].str.contains('cov2'))|
                (df['check_for_covid'].str.contains('ncov'))]
    return(cleaned)

metadata = metadata.fillna('empty')
#adding the title to the abstract so i dont have to seperately check both the abstract and the title
#lowercasing as well
metadata["check_for_covid"] = metadata["abstract"].str.lower() + metadata["title"].str.lower()
metadata = get_relevant(metadata)
metadata = metadata.drop_duplicates(subset='title', keep="first")
del metadata["check_for_covid"]

#A part of my approach is focused on abstracts
#Observations without abstracts are not usefull
#I am removing the texts with no abstracts

metadata = metadata[metadata['abstract'] != 'empty']
len(metadata)
#Output: 64651

#Next I focus on the language of the papers
#Most NLP techniques exceel on English language
#I aim for papers containing english words.

#To extract the language of each paper I use spacy.
#It is written for language processing of scientific articles
#Ref: https://spacy.io/universe/project/spacy-langdetect

#Picking a model designed for mining scientific papers
#The model was selected from:
#https://allenai.github.io/scispacy/
#https://github.com/allenai/scispacy

#Initializing the pipeline of the used model
nlp = en_core_sci_lg.load()

#Adding a Language Detector
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

#A Small demonstration of the usage of the language detector added on the model's pipeline:
test = nlp('Hey Everyone')
test1 = nlp("Mi nombre es Filotas")
if (test1._.language['language'] == "en"):
    print(True, 'It is English')
else:
    print(test1._.language['language'])
#Output : es

if (test._.language['language'] == "en"):
    print(True)
else:
    print(test._.language['language'])
#Output: True

def check_language(abstract):
    abstract_nlp = nlp(abstract)
    language = abstract_nlp._.language['language']
    return(language)

#Applying the model on the Abstracts was time-consuming
#After considering the perfromance of the language detector on a single sentence:
#Applying it on the titles
metadata['Language'] = metadata.apply(lambda row: check_language(str(row['title'])) , axis = 1)

#Checking Language distribtuion
languages = metadata.groupby('Language').agg('count').reset_index()[['Language','title']].sort_values('title')
plt.figure(figsize = (12,8))
plt.barh(languages['Language'],languages['title'])
plt.show()

#English is the most dominant language with a small number of spanish and french articles
#Keeping only the english articles

metadata = metadata[metadata['Language'] == 'en']
del metadata['Language']

#Dropping some duplicates:
metadata = metadata.drop_duplicates()
len(metadata)
#Output 62330

#A total of 62.330 Articles!
#A big dataset concerning the length of each paper (very high dimensional dataset)

#I need to reduce the number of papers even more so the RAM does not crash.
#I make a strong assumption:
#Research regarding Covid is assumed to be splitted into two parts!

#First part regards the outbreak of the virus in Western World(Italy) (January)
#Uncertainty was extremely high till that point and researchers focused on identyfing the mechanisms of the virus
#This period lasted until lockdown guidelines were lifted at May (more or less)

#Then the next period is after May where researchers focused on tackling and stopping the spread of the virus

#These assumptions might be very strong, but to tackle memory issues raised by my computer(Even Colab or Spark FAILED)
#I will split the dataset into two periods.
#High uncertainty period (January - May)
#After-Lockdown period (May - September)

check_dates = [metadata.iloc[i]['publish_time'].month for i in range(len(metadata))]
plt.figure(figsize=(16,8))
plt.hist(check_dates)
plt.show()

#A very big proportion of papers (nearly 30.000) is published on February ( High uncertainty period)
#After that the papers are evenly distributed across the months

metadata['month_of_publish'] = check_dates
very_early_papers = metadata[metadata['month_of_publish'] <= 2]
mid_period_papers = metadata[(metadata['month_of_publish'] > 2) & (metadata['month_of_publish'] <=5 ) ]
late_papers = metadata[metadata['month_of_publish'] > 5]
print("Total of Early Papers:" , len(very_early_papers))
print("Total of Medium-Period Papers:" , len(mid_period_papers))
print("Total of Late Papers:" , len(late_papers))
#Outputs:
# Total of Early Papers: 33473
# Total of Medium-Period Papers: 10573
# Total of Late Papers: 18284

# Finaly some papers had their title as their abstract
# These are also removed as I am interested on the abstracts.
metadata['title'] = metadata['title'].str.lower()
metadata['abstract'] = metadata['abstract'].str.lower()
metadata = metadata[metadata['title']!=metadata['abstract']]

#Saving
filepath = 'D:/complex data project'
name = os.path.join(filepath , 'step1.csv')
metadata.to_csv(name)

