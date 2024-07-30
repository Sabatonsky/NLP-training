# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""
import pandas as pd
import numpy as np
import nltk
import requests
import io

from nltk import word_tokenize
nltk.download('punkt')

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object)

print(df.head())

word2idx = {}
idx2word = []
tokenized_docs = []

for i in range(len(df['text'])):
  words = word_tokenize(df.loc[i,'text'])
  doc_as_int = []
  for word in words:
    if word not in word2idx:
      idx2word.append(word)
      word2idx[word] = len(idx2word) - 1
    doc_as_int.append(word2idx[word])
  tokenized_docs.append(doc_as_int)

N = len(df['text'])
V = len(word2idx)

tf = np.zeros((N, V))

for i, doc_as_int in enumerate(tokenized_docs):
  for j in doc_as_int:
    tf[i, j] += 1

document_freq = np.sum(tf > 0, axis = 0)
idf = np.log(N / document_freq)

tf_idf = tf * idf

i = np.random.choice(N)
row = df.iloc[i]
print("Label:", row['labels'])
print("Text:", row['text'].split("\n", 1)[0])
print("Top 5 terms:")
scores = tf_idf[i]
indices = (-scores).argsort()
for j in indices[:5]:
  print(idx2word[j])
