#Simply reading the json files containing Covid-19 literature (Until Summer 2020)

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from tqdm import tqdm
import joblib


# Loading the saved files
# Download Link :  https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge 

dirs = 'D:/complex data project/pdf_json'
filenames = os.listdir(dirs)

print("Number of articles:", len(filenames))


# The papers could not be appended on a CSV in a single itteration due to RAM issues
# As a result, the following procedure is repeated for a total of 5 times
# The is to split the files into 5 batches and load 5 difference csv files
# The following is the procedure for the 4th (out of 5) batch

papers4 = []
#counting the number of itterations
splits = int(len(filenames)/5)

#1st batch

for i in tqdm(range(4*splits,5*splits)):
    #Adding a slash cuz they dont accept it
    #try:
        #dirs = dirs + "/"
        #merging path
        file = filenames[i]
        file = dirs + "/" + file
        paper = json.load(open(file, 'rb'))

        #Break for testing
        #break
        papers4.append(paper)
        #i = i+1
        #print(i)
    #except:
        #continue


# **Features Extracted**
# The paper_id and the body_text are the only features needed.
# The rest are found on the metadata csv (included in the provided link)

#A function to extract the whole body text:

def get_text(whole_body_text):
    #Taking the paragraphs
    text = [paragraph['text'] for paragraph in whole_body_text]
    #Joining them
    #!!!!!!!!!!!!!!!!!!!!
    #Might change the joining function here and instead of empty lines add a single gap
    full_text = " ".join(text)
    return(full_text)

#Testing its usage
#get_text(test2)

#Function to extract paper_id and body_text
def extraction(paper):
    extracted_features = [paper['paper_id'],get_text(paper['body_text'])]
    #print(len(extracted_features))
    return(extracted_features)

def get_pd(papers):
    features = [extraction(paper) for paper in papers]
    col_names = ["paper_id","text"]

    #Preparing the df
    full_text_df = pd.DataFrame(features,columns=col_names)
    return(full_text_df)




#Running
df5 = get_pd(papers4)

#Saving
filepath = 'D:/complex data project'
name = os.path.join(filepath , 'paper5.csv')
df5.to_csv(name)

#Repeating for the next batch....
#Same exact procedure
dirs = 'D:/complex data project/pmc_json'
filenames = os.listdir(dirs)
print("Number of articles:", len(filenames))
splits = int(len(filenames)/5)

