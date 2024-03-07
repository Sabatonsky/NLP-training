# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:51:26 2024

@author: Bannikov Maxim
"""

import nltk
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

dir_path = r"D:\Training_code"
os.chdir(dir_path)
titles = [line.rstrip() for line in open('all_book_titles.txt')]

stopwords = set(w.rstrip() for  w in open('stopwords.txt'))

stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

for title in titles:
    all_titles.append(title) #Save title in string format and ascii encoding.
    tokens = nltk.tokenize.word_tokenize(title)
    tokens = my_tokenizer(title) #Tokenize all words in title
    all_tokens.append(tokens) #Add to list title but in tokenized format
    for token in tokens:
        if token not in word_index_map: 
            word_index_map[token] = current_index #Add word to D dimention
            current_index += 1 #Move to next index
            index_word_map.append(token) #Backward mapping

def tokens_to_vector(tokens): #Title to count vector
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1
    
svd = TruncatedSVD()
Z = svd.fit_transform(X)

plt.scatter(Z[:,0], Z[:,1])
for i in range(D):
    plt.annotate(text=index_word_map[i], xy=(Z[i,0], Z[i,1]))
plt.show()