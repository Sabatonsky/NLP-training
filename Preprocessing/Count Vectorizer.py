# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
import io

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object)
inputs = df['text']
labels = df['labels']

inputs_train, inputs_test, Y_train, Y_test = train_test_split(inputs, labels)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(inputs_train) #Векторайзер набивает тренировочную матрицу словами и считает их.
X_test = vectorizer.transform(inputs_test) #Векторайзер очищает тренировочную матрицу и считает теперь данные из сета для тестирования. Новые столбцы не создаются. Новые слова игнорируются

model = MultinomialNB()
model.fit(X_train, Y_train)
print('train score:', model.score(X_train, Y_train))
print('test score:', model.score(X_test, Y_test))

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(inputs_train) #Векторайзер набивает тренировочную матрицу словами и считает их.
X_test = vectorizer.transform(inputs_test) #Векторайзер очищает тренировочную матрицу и считает теперь данные из сета для тестирования. Новые столбцы не создаются. Новые слова игнорируются

model = RandomForestClassifier(200)
model.fit(X_train, Y_train)
print('train score:', model.score(X_train, Y_train))
print('test score:', model.score(X_test, Y_test))

def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith("J"):
    return wordnet.ADJ
  elif treebank_tag.startswith("V"):
    return wordnet.VERB
  elif treebank_tag.startswith("N"):
    return wordnet.NOUN
  elif treebank_tag.startswith("R"):
    return wordnet.ADV
  else:
    return wordnet.NOUN

class LemmaTokenizer:
  def __init__(self):
    self.wnl = WordNetLemmatizer() #Функция nltk, которая превращает слова в леммы
  def __call__(self, doc):
    tokens = word_tokenize(doc) #Генерим токены из документа
    words_and_tags = nltk.pos_tag(tokens) #Определяем часть речи и объединяем со словом, делаем лист из них
    return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in words_and_tags] #Используя WordNetLemmatizer генерим лист, в котором слова лематизируются в завизимости от их части речи.

vectorizer = CountVectorizer(tokenizer=LemmaTokenizer()) #Подгружаем список лем в countvectorizer,
X_train = vectorizer.fit_transform(inputs_train)
X_test = vectorizer.transform(inputs_test)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('train score:', model.score(X_train, Y_train))
print('test score:', model.score(X_test, Y_test))

class StemTokenizer:
  def __init__(self):
    self.ps = PorterStemmer()
  def __call__(self, doc):
    tokens = word_tokenize(doc)
    return [self.ps.stem(t) for t in tokens]

vectorizer = CountVectorizer(tokenizer=StemTokenizer()) #Подгружаем список стволов в countvectorizer,
X_train = vectorizer.fit_transform(inputs_train)
X_test = vectorizer.transform(inputs_test)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('train score:', model.score(X_train, Y_train))
print('test score:', model.score(X_test, Y_test))
