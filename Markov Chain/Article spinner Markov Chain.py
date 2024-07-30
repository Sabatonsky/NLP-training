# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:32:28 2023

@author: Bannikov Maxim
"""

import pandas as pd
import numpy as np
import nltk
import requests
import io
import textwrap

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object)

label = 'business'
texts = df[df['labels'] == label]['text']
texts.head()

probs = {}

for doc in texts:
    lines = doc.split('\n')
    for line in lines:
        tokens = word_tokenize(line)
        for i in range(len(tokens) - 2):
            t_0 = tokens[i]
            t_1 = tokens[i + 1]
            t_2 = tokens[i + 2]
            key = (t_0, t_2)
            if key not in probs:
                probs[key] = {}
            if t_1 not in probs[key]:
                probs[key][t_1] = 1
            probs[key][t_1] += 1

for key, d in probs.items():
    total = sum(d.values())
    for k, v in d.items():
        d[k] = v / total

def spin_document(doc):
    lines = doc.split('\n')
    output = []
    for line in lines:
        if line:
            new_line = spin_line(line)
        else:
            new_line = line
        output.append(new_line)
    return '\n'.join(output)

detokenizer = TreebankWordDetokenizer()

detokenizer.detokenize(word_tokenize(texts.iloc[0].split("\n")[2]))

def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t

def spin_line(line):
    tokens = word_tokenize(line)
    output = [tokens[0]]
    i = 0
    while i < len(tokens) - 2:
        t_0 = tokens[i]
        t_1 = tokens[i + 1]
        t_2 = tokens[i + 2]
        key = (t_0, t_2)
        p_dist = probs[key]
        if len(p_dist) > 1 and np.random.random() > 0.3:
            middle = sample_word(p_dist)
            output.append(t_1)
            output.append("<" + middle + ">")
            output.append(t_2)
            i += 2
        else:
            output.append(t_1)
            i += 1
    if i == len(tokens) - 2:
        output.append(tokens[-1])
    return detokenizer.detokenize(output)

i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]
new_doc = spin_document(doc)
        
print(textwrap.fill(new_doc, replace_whitespace = False, fix_sentence_endings=True))
