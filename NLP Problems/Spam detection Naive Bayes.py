# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import requests
import io
import nltk
nltk.download('punkt')

response = requests.get('https://lazyprogrammer.me/course_files/spam.csv')
file_object = io.StringIO(response.content.decode('ISO-8859-1'))

df = pd.read_csv(file_object)
df.drop('Unnamed: 2', axis = 1, inplace = True)
df.drop('Unnamed: 3', axis = 1, inplace = True)
df.drop('Unnamed: 4', axis = 1, inplace = True)
df.rename(columns = {'v2': 'text', 'v1': 'labels'}, inplace = True)
df['labels'] = df['labels'].map({'ham':0, 'spam':1})
inputs = df['text']
labels = df['labels']

inputs_train, inputs_test, Y_train, Y_test = train_test_split(inputs, labels)
vectorizer = TfidfVectorizer(decode_error = 'ignore')
X_train = vectorizer.fit_transform(inputs_train)
X_test = vectorizer.transform(inputs_test)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('train score:', model.score(X_train, Y_train))
print('test score:', model.score(X_test, Y_test))

Prob_train = model.predict_proba(X_train)[:, 1]
Prob_test = model.predict_proba(X_test)[:, 1]
print('train AUC:', roc_auc_score(Y_train, Prob_train))
print('test AUC:', roc_auc_score(Y_test, Prob_test))

P_train = model.predict(X_train)
P_test = model.predict(X_test)
print('train f1:', f1_score(Y_train, P_train))
print('test f1:', f1_score(Y_test, P_test))

cm = confusion_matrix(Y_train, P_train)
cm

def plot_cm(cm):
  classes = ['ham', 'spam']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sn.heatmap(df_cm, annot=True, fmt='g')
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Target')

plot_cm(cm)

cm_test = confusion_matrix(Y_test, P_test)
plot_cm(cm_test)

def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['text']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width = 600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

visualize(1)

visualize(0)

X = vectorizer.transform(df['text'])
df['predictions'] = model.predict(X)

sneaky_spam = df[(df['predictions'] == 0) & (df['labels'] == 1)]['text']
for msg in sneaky_spam:
  print(msg)

not_actually_spam = df[(df['predictions'] == 1) & (df['labels'] == 0)]['text']
for msg in not_actually_spam:
  print(msg)