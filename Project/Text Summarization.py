# Databricks notebook source
#upgrade pip
%pip install --upgrade pip

# COMMAND ----------

#install spacy, textblob
import sys
!{sys.executable} -m pip install spacy
!{sys.executable} -m spacy download en_core_web_sm
!{sys.executable} -m pip install textblob

# COMMAND ----------

#import methods from textblob, nltk, numpy,spacy
from textblob import Word
from textblob import TextBlob
import numpy as np
import spacy

# COMMAND ----------

#Read reviews data from aws s3 bucket
review_data = spark.read.option("header","true").csv("s3://assignment2.a/Reviews.csv")
#Retrieve Text column from the reviews dataset
review_text=review_data.select("Text")
display(review_text)

# COMMAND ----------

#Convert the data in the text column to lowercase and split the data based on fullstop(. )
df = review_text.rdd.map(lambda x: list(x)).map(lambda x: (str(x[0]).split('. '),str(x[0]).lower().split('. ')))
df.take(10)

# COMMAND ----------

#Function for removal of special characters and numerals
def remove_special_characters(review):
    temp_list = []
    str =''
    for sentence in review:
        str += re.sub(r"[^a-zA-Z \n\']","",sentence)
        temp_list.append(str)
        str =''
    return temp_list


# COMMAND ----------

import re
#Apply remove_special_characters function on the modified text column
df_remove_special_characters = df.map(lambda x: (x[0],remove_special_characters(x[1])))
df_remove_special_characters.take(5)

# COMMAND ----------


nlp = spacy.load("en_core_web_sm")

#Function for convertion of the words to their base form(lemmatization)
def lemmatization(reviews):
    temp_list =[]
    for string in reviews:
        
        temp_list.append(" ".join([token.lemma_.lower() for token in nlp(string)]))
    return temp_list

# COMMAND ----------

#Apply lemmatization function on the modified text column
df_lemmatization = df_remove_special_characters.map(lambda x: (x[0],lemmatization(x[1])))
df_lemmatization.take(5)

# COMMAND ----------

#Function for typo correction 
def correct_sentence_spelling(sentence):    
    sentence = TextBlob(sentence)
    result = sentence.correct()
    return result

def typos_correction(review):
    temp_list = []
    for sentence in review:
        temp_list.append(str(correct_sentence_spelling(sentence)))
    return temp_list

# COMMAND ----------

#Apply correct_sentence_spelling function on the modified text column
df_typos_correction = df_lemmatization.map(lambda x: (x[0],typos_correction(x[1])))
df_typos_correction.take(5)

# COMMAND ----------

#Function to split sentences to words
def split_words(review):
    temp_list = []
    for sentence in review:
        temp_list.append(sentence.split(' '))
    return temp_list

# COMMAND ----------

#Apply split_words function on the modified text column
df_split_words = df_typos_correction.map(lambda x: (x[0],split_words(x[1])))
df_split_words.take(5)

# COMMAND ----------

#Function to remove stop words and white spaces
def stopword_removal(review):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    temp_list = []
    for j in range(len(review)):
        temp_list1 = []
        for word in review[j]:
            if word not in stop_words and len(word) != 0:
                temp_list1.append(word)
        temp_list.append(temp_list1)
    return temp_list

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
df_stopword_removal = df_split_words.map(lambda x: (x[0],stopword_removal(x[1])))
df_stopword_removal.take(5)

# COMMAND ----------

#Function to find the similarity between sentences
def similarity_matrix(review):
    #matrix creation
    rows = len(review)
    similarity = np.zeros((rows, rows), dtype='float')
    for j in range(len(review)):
        for k in range(len(review)):
            result = [word for word in review[j] if word in review[k]]
            #sentence weight computation
            if(len(review[j]) * len(review[k]))!=0:
                similarity[j][k] = len(result) / (len(review[j]) * len(review[k]))
    return similarity

# COMMAND ----------

#Apply similarity_matrix function on the modified text column
df_similarity =df_stopword_removal.map(lambda x: (x[0],similarity_matrix(x[1])))
df_similarity.take(5)

# COMMAND ----------

#function to normalize the similarity matrix
def symmetrize(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())

def norm(matrix):
    graph_temp = symmetrize(matrix)
    
    norm = np.sum(matrix, axis=0)
    g_norm = np.divide(matrix, norm, where=norm!=0) 
    return g_norm

# COMMAND ----------

#Apply norm function on the matrix found in previous step
df_norm = df_similarity.map(lambda x: (x[0],norm(x[1])))
df_norm.take(5)

# COMMAND ----------

#Function to compute the text rank using the formula
def text_rank(matrix):
    damping_factor = 0.85
    tr = np.array([1] * len(matrix))
    computed_tr = 0
    for i in range(10):
        tr = (1-damping_factor) + damping_factor * np.dot(matrix, tr)
        if abs(computed_tr - sum(tr))  < 0.00001:
            break
        else:
            computed_tr = sum(tr)
    return tr

# COMMAND ----------

#Apply text_rank function on the matrix found in previous step
df_text_rank = df_norm.map(lambda x: (x[0],text_rank(x[1])))
df_text_rank.take(5)

# COMMAND ----------

#Function to map sentences with their respective text ranks
def sentence_ranking(text_rank_array, review):
    node_weight={}
    for i in range(len(review)):
        node_weight[review[i]] = text_rank_array[i]
    return node_weight

# COMMAND ----------

import math
from math import ceil

#Apply sentence_ranking function to get a dictionary of sentences and their text ranks for all reviews
df_mapping = df_text_rank.map(lambda x: sentence_ranking(x[1],x[0]) )
df_mapping.take(5)

# COMMAND ----------

#Function to get the extractive summary of the review based on the text rank of the sentences
def text_summarization(dictionary):
    #display n/2 sentences with top text rank in each review
    no_of_sentences = ceil(len(dictionary)/2)
    return list(dict(sorted(dictionary.items(), key=lambda item: item[-1], reverse=True)[:no_of_sentences]).keys())

# COMMAND ----------

#Apply text_summarization function to get the extractive summary of the reviews
df_text_summarization = df_mapping.map(lambda x: text_summarization(x)).map(lambda x: ' '.join(x))
df_text_summarization.take(5)

# COMMAND ----------

from pyspark.sql.types import StringType
#Convertion of the rdd to dataframe and display of text summary
text_summarization = spark.createDataFrame(df_text_summarization,StringType()).withColumnRenamed("value", "Review Summary")
display(text_summarization)

# COMMAND ----------

from functools import reduce
df_mapping = df_text_rank.map(lambda x: sentence_ranking(x[1],x[0]) )
result_dict = reduce(lambda a, b: {**a, **b},df_mapping.take(3))

#Line plot for 3 reviews (rank vs sentences)
import matplotlib.pyplot as plt
y = result_dict.values()
x = result_dict.keys()

plt.plot(x,y)
plt.ylabel('Rank')
plt.xlabel('Sentences')

# COMMAND ----------

# MAGIC %pip install rouge

# COMMAND ----------

#Evaluating ROUGE metrics(precision, recall, F-measure) for one review in the dataset
from rouge import Rouge
#take extractive summary of one review from the built model
model_out = ['Great taffy at a great price.  There was a wide assortment of yummy taffy']
#take reference summary from the data set for the same review
reference =['Great taffy collection']
rouge = Rouge()
rouge.get_scores(model_out, reference, avg=True)

# COMMAND ----------


