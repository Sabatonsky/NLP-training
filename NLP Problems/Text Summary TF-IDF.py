# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import io
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object)
text = df.loc[df.labels == 'business',"text"].sample()
sent_list = sent_tokenize(text.iloc[0].split("\n", 1)[1])

def wrap(x):
  return textwrap.fill(x, replace_whitespace=True, fix_sentence_endings=True)

def get_score(tfidf_row):
  x = tfidf_row[tfidf_row != 0]
  return x.mean()

def summarize(text):
  vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), norm = 'l1')
  text_vec = vectorizer.fit_transform(sent_list)
  text_size = text_vec.shape[0]
  s_size = int(0.3*text_size)
  scores = np.array([get_score(text_vec[i, :]) for i in range(text_size)])
  sort_idx = np.argsort(-scores)

  for i in sort_idx[:s_size + 1]:
    print(wrap("%.2f: %s" % (scores[i], sent_list[i])))

summarize(sent_list)
