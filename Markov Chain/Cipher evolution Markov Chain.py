# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:42:58 2023

@author: Bannikov Maxim
""" 

import numpy as np
import nltk
import requests
import io
import string
import regex as re
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def gen_cipher():
    letters = list(string.ascii_lowercase)
    np.random.shuffle(letters)
    return letters

def cipher_encrypt(text, cipher):
    letters = list(string.ascii_lowercase)
    cipher_full = dict(zip(letters, cipher))
    output = ""
    for l in text:
        output += cipher_full.get(l, l)
    return output

def cipher_decrypt(text, cipher):
    letters = list(string.ascii_lowercase)
    cipher_full = dict(zip(cipher, letters))
    output = ""
    for l in text:
        output += cipher_full.get(l, l)
    return output

def mutate(cipher):
    l_0, l_1 = np.random.choice(range(len(cipher)), 2, replace=False)
    new_cipher = cipher.copy()
    new_cipher[l_0], new_cipher[l_1] = new_cipher[l_1], new_cipher[l_0]
    return new_cipher

def score(text, cipher, bg, ug):
    dc_text = cipher_decrypt(text, cipher)
    words = word_tokenize(dc_text)
    output = 0
    letters = list(string.ascii_lowercase)
    V = len(string.ascii_lowercase)
    mapping = dict(zip(letters, range(V)))
    for word in words:
        l_0 = mapping[word[0]]
        output += ug[l_0]
        for l in range(len(word) - 1):
            l_0 = mapping[word[l]]
            l_1 = mapping[word[l + 1]]
            output += bg[l_0, l_1]
    return output

if __name__ == "__main__":
    
    text = '''I then lounged down the street and found,
    as I expected, that there was a mews in a lane which runs down
    by one wall of the garden. I lent the ostlers a hand in rubbing
    down their horses, and received in exchange twopence, a glass of
    half-and-half, two fills of shag tobacco, and as much information
    as I could desire about Miss Adler, to say nothing of half a dozen
    other people in the neighbourhood in whom I was not in the least
    interested, but whose biographies I was compelled to listen to.
    '''
    regex = re.compile('[^a-zA-Z]')  
    text = text.rstrip().lower()  
    text = regex.sub(' ', text)
    m_cipher = gen_cipher()
    enc_text = cipher_encrypt(text, m_cipher)
    
    url1 = "https://lazyprogrammer.me/course_files/moby_dick.txt"
    
    r=requests.get(url1)   
    file_object = io.StringIO(r.content.decode('utf-8'))
    s = file_object.getvalue().rstrip().lower()  
    s = regex.sub(' ', s)
    tokens = word_tokenize(s)
    V = len(string.ascii_lowercase)
    letters = list(string.ascii_lowercase)
    ug = np.ones(V)    
    bg = np.ones((V, V))
    mapping = dict(zip(letters, range(V)))
        
    for word in tokens:
        l_0 = mapping[word[0]]
        ug[l_0] += 1
        for l in range(len(word) - 1):
            l_0 = mapping[word[l]]
            l_1 = mapping[word[l + 1]]
            bg[l_0, l_1] += 1
            
    ug /= ug.sum()
    ug = np.log(ug)
    bg /= bg.sum(axis = 1, keepdims = True)
    bg = np.log(bg)
    
    parent_list = []
    
    for i in range(20):  
        parent_list.append(gen_cipher())
        
    for epoch in range(3001):
        parent_scores = []
        child_scores = []
        child_list = []
        for p in parent_list:
            parent_score = score(enc_text, p, bg, ug)
            parent_scores.append(parent_score)
            for i in range(3):
                child = mutate(p)
                child_list.append(child)
                child_score = score(enc_text, child, bg, ug)
                child_scores.append(child_score)
        top_parent = np.argsort(parent_scores)[-5:]
        top_child = np.argsort(child_scores)[-15:]
        parent_list = [parent_list[i] for i in top_parent]
        child_list = [child_list[i] for i in top_child]
        parent_list += child_list.copy()
        
        if epoch % 100 == 0:
            parent_scores = []
            for p in parent_list:
                parent_score = score(enc_text, p, bg, ug)
                parent_scores.append(parent_score)
            top_index = np.argmax(parent_scores)
            top_score = np.max(parent_scores)
            top_cipher = parent_list[top_index]
            correct = 0
            for i in range(26):
                if top_cipher[i] == m_cipher[i]:
                    correct += 1
            accuracy = round(correct / 26, 2)
            print('leading cipher:', top_cipher)
            print('top score:', top_score)
            print('accuracy:', accuracy)
            
score(enc_text, m_cipher, bg, ug)
cipher_decrypt(enc_text, top_cipher)
