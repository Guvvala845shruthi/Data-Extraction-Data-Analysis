#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


# In[2]:


def count_syllables(word):
    vowels = 'aeiou'
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


# In[3]:


def complex_word_percentage(text, stopwords):
    words = word_tokenize(text)
    word_count = len(words)
    complex_word_count = sum(1 for word in words if count_syllables(word) > 2 and word.lower() not in stopwords)
    return (complex_word_count / word_count) * 100


# In[4]:


def fog_index(average_sentence_length, complex_word_percentage):
    return 0.4 * (average_sentence_length + complex_word_percentage)


# In[5]:


def sentimental_analysis(text, positive_words, negative_words, stopwords):
    words = word_tokenize(text)
    positive_score = sum(1 for word in words if word.lower() in positive_words and word.lower() not in stopwords)
    negative_score = sum(1 for word in words if word.lower() in negative_words and word.lower() not in stopwords)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score


# In[23]:


positive_words_path = r"C:\Users\Home\Desktop\positive-words.txt"
positive_words = set(open(positive_words_path).read().splitlines())


# In[22]:


import os

positive_words_path = r"C:\Users\Home\Desktop\positive-words.txt"

if os.path.exists(positive_words_path):
    positive_words = set(open(positive_words_path).read().splitlines())
else:
    print(f"File not found: {positive_words_path}")


# In[24]:


negative_words_path = r"C:\Users\Home\Desktop\negative-words.txt"
negative_words = set(open(negative_words_path).read().splitlines())


# In[25]:


input_data = pd.read_excel(r"C:\Users\Home\Desktop\Input.xlsx")


positive_words = set(open(positive_words_path).read().splitlines())
negative_words = set(open(negative_words_path).read().splitlines())


# In[28]:


nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('English'))


# In[29]:


pip install nltk


# In[30]:


pip install spacy


# In[31]:


python -m spacy download en_core_web_sm


# In[32]:


import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "This is a sample sentence. spaCy is an alternative to NLTK."

# Process the text with spaCy
doc = nlp(text)

# Print tokenized words
for token in doc:
    print(token.text)


# In[33]:


import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Get the set of spaCy English stopwords
stopwords = set(nlp.Defaults.stop_words)

# Example: Print the spaCy stopwords
print(stopwords)


# In[34]:


output_data = []


# In[56]:


for _, row in input_df.iterrows():
    url_id = row['URL_ID']
    file_path = os.path.join("articles", f"{url_id}.txt")

    with open(file_path, 'r', encoding='utf-8') as file:
        article_text = file.read()

    # Sentimental 
    positive_score, negative_score, polarity_score, subjectivity_score = sentimental_analysis(article_text,
                                                                                                positive_words,
                                                                                                negative_words,
                                                                                                stopwords)

    # Readability
    sentences = sent_tokenize(article_text)
    total_words = len(word_tokenize(article_text))
    total_sentences = len(sentences)

    # zero case for average_sentence_length and average_words_per_sentence

    if total_sentences == 0:
        average_sentence_length = 0
        average_words_per_sentence = 0
    else:
        average_sentence_length = total_words / total_sentences
        average_words_per_sentence = total_words / total_sentences
        # zero case for complex_word_percentage

    if total_words == 0:
        percentage_complex_words = 0
    else:
        percentage_complex_words = complex_word_percentage(article_text, stopwords)

    fog_index_score = fog_index(average_sentence_length, percentage_complex_words)

    # Complex Word Count

    complex_word_count = sum(1 for word in word_tokenize(article_text) if count_syllables(word) > 2)

    # Word Count

    word_count = len(re.findall(r'\b\w+\b', article_text))

    # zero case for syllable_per_word

    if word_count == 0:
        syllable_per_word = 0
    else:
        total_syllables = sum(count_syllables(word) for word in word_tokenize(article_text))
        syllable_per_word = total_syllables / word_count

    # Personal Pronouns

    personal_pronouns_count = len(re.findall(r'\b(I|we|my|ours|us)\b', article_text, flags=re.IGNORECASE))

    # zero case for average_word_length
     if len(word_tokenize(article_text)) == 0:
        average_word_length = 0
    else:
        words = word_tokenize(article_text)
        total_characters = sum(len(word) for word in words)
        average_word_length = total_characters / len(words)




    output_data.append([url_id, positive_score, negative_score, polarity_score, subjectivity_score,
                        average_sentence_length, percentage_complex_words, fog_index_score,
                        average_words_per_sentence, complex_word_count, word_count,
                        syllable_per_word, personal_pronouns_count, average_word_length])





output_df = pd.DataFrame(output_data, columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                                               'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                                               'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                                               'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',
                                               'AVG WORD LENGTH'])


# In[60]:


import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Assuming you have the necessary functions and data loaded already

output_data = []

# Assuming input_df is your DataFrame
for _, row in input_data.iterrows():
    url_id = row['URL_ID']
    file_path = os.path.join(os.getcwd(), f"{url_id}.txt")  # Assuming text files are in the same directory

    # Check if the file exists before attempting to open it
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            article_text = file.read()

        # Rest of your code...

    else:
        print(f"File not found for URL_ID: {url_id}")

output_df = pd.DataFrame(output_data, columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                                               'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                                               'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                                               'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',
                                               'AVG WORD LENGTH'])


# In[63]:


output_df.to_excel("Output Data Structure.xlsx", index=False)


# In[ ]:




