# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import pandas as pd
import requests
import numpy as np
import string
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

url1 = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt"
url2 = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt"

s=requests.get(url1).text.rstrip().lower()
s = s.translate(str.maketrans('', '', string.punctuation))
s = s.split('\n')
df_eap = pd.DataFrame(s)
df_eap.rename(columns = {0: 'inputs'}, inplace = True)
df_eap['labels'] = 0

s=requests.get(url2).text.rstrip().lower()
s = s.translate(str.maketrans('', '', string.punctuation))
s = s.split('\n')
df_rf = pd.DataFrame(s)
df_rf.rename(columns = {0: 'inputs'}, inplace = True)
df_rf['labels'] = 1

df = pd.concat([df_rf, df_eap], ignore_index = True, axis = 0)

inputs_train, inputs_test, Y_train, Y_test = train_test_split(df["inputs"], df["labels"])
word2idx = {'<unk>':0}
idx2word = ['<unk>']
for input in inputs_train:
  words = word_tokenize(input)
  for word in words:
    if word not in idx2word:
      word2idx[word] = len(idx2word)
      idx2word.append(word)

train_input_int = []
test_input_int = []

for input in inputs_train:
  tokens = word_tokenize(input)
  line_as_int = [word2idx[token] for token in tokens]
  train_input_int.append(line_as_int)

for input in inputs_test:
  tokens = word_tokenize(input)
  line_as_int = [word2idx.get(token, 0) for token in tokens]
  test_input_int.append(line_as_int)

V = len(word2idx)
A0 = np.ones((V, V))
pi0 = np.ones(V)

A1 = np.ones((V, V))
pi1 = np.ones(V)

def compute_counts(text_as_int, A, pi):
  for tokens in text_as_int:
    if tokens:
      pi[tokens[0]] += 1
      for i in range(1, len(tokens)):
        A[tokens[i - 1], tokens[i]] += 1
  return A, pi

A0, pi0 = compute_counts([t for t, y in zip(train_input_int, Y_train) if y == 0], A0, pi0)
A1, pi1 = compute_counts([t for t, y in zip(train_input_int, Y_train) if y == 1], A1, pi1)

A0 /= A0.sum(axis=1, keepdims = True)
pi0 /= pi0.sum()
A1 /= A1.sum(axis=1, keepdims = True)
pi1 /= pi1.sum()

logA0 = np.log(A0)
logpi0 = np.log(pi0)
logA1 = np.log(A1)
logpi1 = np.log(pi1)

count0 = sum(Y_train == 0)
count1 = sum(Y_train == 1)
total = len(Y_train)
p0 = count0 / total
p1 = count1 / total
logp0 = np.log(p0)
logp1 = np.log(p1)
p0, p1

class Classifier:
  def __init__(self, logAs, logpis, logpriors):
    self.logAs = logAs
    self.logpis = logpis
    self.logpriors = logpriors
    self.K = len(logpriors)

  def compute_log_likelihood(self, input_, class_):
    logA = self.logAs[class_]
    logpi = self.logpis[class_]
    logprob = 0
    logprob += logpi[0]
    for i in range(1, len(input_)):
      logprob += logA[input_[i - 1], input_[i]]
    return logprob #Так как вероятности логарифмированы, мы их не умножаем, а суммируем. Сначала набиваем данные из p, так как это первое слово.
    #Затем берем данные из A в зависимости от текущего слова и предыдущего

  def predict(self, inputs):
    predictions = np.zeros(len(inputs))
    for i, input_ in enumerate(inputs):
      posteriors = [self.compute_log_likelihood(input_, c) + self.logpriors[c] for c in range(self.K)] #Высчитываем вероятности и добавляем log prior вероятность: распределение категорий в prior distribution.
      pred = np.argmax(posteriors)
      predictions[i] = pred
    return(predictions) #

clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])
Ptrain = clf.predict(train_input_int)
print('Train acc:', np.mean(Ptrain == Y_train))

Ptest = clf.predict(test_input_int)
print('Test acc:', np.mean(Ptest == Y_test))
