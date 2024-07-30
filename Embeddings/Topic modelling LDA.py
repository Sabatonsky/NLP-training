# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap
import requests
import io

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#TF-IDF analog is implemented in LDA through beta hyperparameter, therefore there is no need to use it. We take ususal CountVectorizer.
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('stopwords')

stops = set(stopwords.words('english'))
stops = stops.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two', 'mr', 'new', 'year'
})

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object)

vectorizer = CountVectorizer(stop_words=list(stops))
X = vectorizer.fit_transform(df['text'])

lda = LatentDirichletAllocation(n_components=10, random_state=12345)
lda.fit(X)

def plot_top_words(model, feature_names, n_top_words=10):
  fig, axes = plt.subplots(2, 5, figsize=(20, 15), sharex=True)
  axes = axes.flatten()
  for topic_idx, topic in enumerate(model.components_):
    top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax=axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize":30})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=20)
    for i in "top right left".split():
      ax.spines[i].set_visible(False)
    fig.suptitle("LDA", fontsize=40)
    plt.savefig("LDA topics")

feature_names = vectorizer.get_feature_names_out()
plot_top_words(lda, feature_names)

Z = lda.transform(X)
np.random.seed(0)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels'])
plt.savefig("LDA labels")

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

print(wrap(df.iloc[i]['text']))

i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels'])
plt.savefig("LDA labels next")

print(wrap(df.iloc[i]['text']))
