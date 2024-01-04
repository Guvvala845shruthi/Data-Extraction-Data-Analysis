#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Read the input data from "Input.xlsx"
input_data = pd.read_csv(r"C:\Users\Home\Desktop\Input.csv", encoding="ISO-8859-1")


# In[2]:


pip install newspaper3k


# In[3]:


import newspaper
import logging

def extract_article_text(url):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()

        # Extract the article title and text
        article_title = article.title
        article_text = article.text.strip()

        return article_title, article_text

    except Exception as e:
        logging.warning(f"Failed to extract text for URL: {url}")
        return "", ""


# In[4]:


# Loop through each URL and extract article text, then save it as a text file
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    article_title, article_text = extract_article_text(url)

    # Save the extracted article text to a text file with URL_ID as its file name
    with open(f'{url_id}.txt', 'w', encoding='utf-8') as file:
        file.write(f'{article_title}\n\n{article_text}')

print("Data extraction and saving completed.")


# In[ ]:





# In[ ]:




