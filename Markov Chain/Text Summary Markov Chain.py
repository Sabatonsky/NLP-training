# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), norm = 'l1')
text_vec = vectorizer.fit_transform(sent_list)
text_size = text_vec.shape[0]
s_size = int(0.3*text_size)

#Solution is based on Perron-Frobenius theorem for Markov matrix.
#Main idea of TextRank is based on idea that there are sentences that are more generalistic. Other explain some minor details.
#If sentence is similar to many other sentences, then it is good in generalization and suitable for summary.
#We measure similarity through proxy - cosine similarity.
scores = cosine_similarity(text_vec)
# Normalization
scores /= scores.sum(axis = 1, keepdims = True)
# U is smoothing, all sentences should have some probability to go to other sentence, even if they don't (PF theorem condition)
U = np.ones_like(scores) / len(scores)
# But we need to make scores sum to 1, so U should be subtracted from total G.
a = 0.15
scores = a*U + (1-a)*scores

eigenvals, eigenvecs = np.linalg.eig(scores.T)

limiting_dist = np.ones(len(scores)) / len(scores)
threshold = 1e-8
delta = float('inf')
iters = 0
while delta > threshold:
  iters += 1
  p = limiting_dist.dot(scores)
  delta = np.abs(p - limiting_dist).sum()
  limiting_dist = p
print(iters)

def wrap(x):
  return textwrap.fill(x, replace_whitespace=True, fix_sentence_endings=True)

sort_idx = np.argsort(-limiting_dist)
for i in sort_idx[:s_size + 1]:
  print(wrap("%.2f: %s" % (limiting_dist[i], sent_list[i])))
