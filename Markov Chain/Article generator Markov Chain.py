# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import pandas as pd
import requests
import numpy as np
import string

import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt"

s=requests.get(url).text.rstrip().lower()
s = s.translate(str.maketrans('', '', string.punctuation))
s = s.split('\n')
df_rf = pd.DataFrame(s)
df_rf.rename(columns = {0: 'inputs'}, inplace = True)

def add2dict(d, k, v):
  if k not in d:
    d[k] = {}
  d[k][v] = d[k].get(v, 0) + 1

texts = []
A_2 = {}
A_1 = {}
A_0 = {}
for line in df_rf['inputs']:
    words = word_tokenize(line)
    T = len(words)
    if T > 1:
      t_2 = words[0]
      A_0[t_2] = A_0.get(t_2, 0) + 1
      t_1 = words[1]
      add2dict(A_1, t_2, t_1)

      for i in range(2, T - 1):
        key = (t_2, t_1)
        t_0 = words[i]
        add2dict(A_2, key, t_0)
        t_2 = words[i - 1]
        t_1 = words[i]

      key = (t_2, t_1)
      add2dict(A_2, key, 'END')

A_0 = {k:v/sum(A_0.values()) for k, v in A_0.items()}

for i in A_1.keys():
  total = sum(A_1[i].values())
  A_1[i] = {k:v/total for k, v in A_1[i].items()}

for i in A_2.keys():
  total = sum(A_2[i].values())
  A_2[i] = {k:v/total for k, v in A_2[i].items()}

def generate(w_l, A_0, A_1, A_2):
  result = []
  for i in range(w_l):
    w_0 = np.random.choice([k for k in A_0.keys()], p = [v for v in A_0.values()])
    k_1 = [k for k in A_1[w_0].keys()]
    v_1 = [v for v in A_1[w_0].values()]
    w_1 = np.random.choice(k_1, p = v_1)
    output = [w_0, w_1]
    while True:
      key = (w_0, w_1)
      k_2 = [k for k in A_2[key].keys()]
      v_2 = [v for v in A_2[key].values()]
      w_2 = np.random.choice(k_2, p = v_2)
      if w_2 == 'END':
        break
      w_0 = w_1
      w_1 = w_2
      output.append(w_2)
    dt_output = TreebankWordDetokenizer().detokenize(output)
    result.append(dt_output)
  return '\n'.join(result)

poem = generate(4, A_0, A_1, A_2)
print(poem)
