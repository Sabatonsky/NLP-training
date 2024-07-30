# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:06:01 2023

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import io
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import TweetTokenizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

url = "https://lazyprogrammer.me/course_files/AirlineTweets.csv"
r=requests.get(url)
file_object = io.StringIO(r.content.decode('utf-8'))
df = pd.read_csv(file_object)

df_short = df.loc[:,('text','airline_sentiment')]
inputs = df_short['text']
labels = df_short['airline_sentiment']

vec = []
for i in inputs:
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    inputs_vec = tknzr.tokenize(i.lower().strip())
    inputs_str = " ".join(inputs_vec)
    vec.append(inputs_str)

input_train, input_test, Y_train, Y_test = train_test_split(vec, labels)
vectorizer = TfidfVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(input_train)
X_test = vectorizer.transform(input_test)

model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, Y_train)
P_test = model.predict(X_test)

print('Train acc:', model.score(X_train, Y_train))
print('Test acc:', model.score(X_test, Y_test))

Pr_train = model.predict_proba(X_train)
Pr_test = model.predict_proba(X_test)

print('Train auc:', roc_auc_score(Y_train, Pr_train, multi_class = 'ovo'))
print('Test auc:', roc_auc_score(Y_test, Pr_test, multi_class = 'ovo'))

P_train = model.predict(X_train)
P_test = model.predict(X_test)

cm = confusion_matrix(Y_train, P_train, normalize='true')

def plot_cm(cm):
    classes = ['negative', 'positive', 'neutral']
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    ax = sn.heatmap(df_cm, annot = True, fmt = 'g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.show()
    
plot_cm(cm)

cm_test = confusion_matrix(Y_test, P_test, normalize='true')    
plot_cm(cm_test)

df_b_short = df_short[df_short.airline_sentiment != 'neutral']
input_train, input_test, Y_train, Y_test = train_test_split(df_b_short['text'], df_b_short['airline_sentiment'])

X_train = vectorizer.fit_transform(input_train)
X_test = vectorizer.transform(input_test)

model1 = LogisticRegression(max_iter=500, class_weight='balanced')
model1.fit(X_train, Y_train)
P_test = model.predict(X_test)

print('Train acc:', model1.score(X_train, Y_train))
print('Test acc:', model1.score(X_test, Y_test))

Pr_train = model1.predict_proba(X_train)[:, 1]
Pr_test = model1.predict_proba(X_test)[:, 1]

print('Train auc:', roc_auc_score(Y_train, Pr_train))
print('Test auc:', roc_auc_score(Y_test, Pr_test))

P_train = model1.predict(X_train)
P_test = model1.predict(X_test)

cm = confusion_matrix(Y_train, P_train, normalize='true')

plt.hist(model1.coef_[0], bins = 30) #Мы анализируем эффект слов на решение. В основном слова нейтральны, но есть несколько, по которым мы делаем вывод
plt.show()

word_index_map = vectorizer.vocabulary_
threshold = 2

print("Most positive words:")
for word, index in word_index_map.items():
    weight = model1.coef_[0][index]
    if weight > threshold:
        print(word, weight)
    
for word, index in word_index_map.items():
    weight = model1.coef_[0][index]
    if weight < -threshold:
        print(word, weight)
        
miss_comments = input_train[P_train != Y_train].reset_index(drop = True)
miss_probs = Pr_train[P_train != Y_train]

model1.classes_ #negative 0, positive 1

pos_false = miss_comments[np.argmax(miss_probs)] #Отзыв оценен как позитивный, но на самом деле он негативный
print(pos_false, np.max(miss_probs))
neg_false = miss_comments[np.argmin(miss_probs)]
print(neg_false, np.min(miss_probs))
