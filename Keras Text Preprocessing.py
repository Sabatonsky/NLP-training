# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:34:57 2024

@author: Bannikov Maxim
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions."
    ]

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

tokenizer.word_index

data = pad_sequences(sequences)
print(data)

data = pad_sequences(sequences, padding='post')
print(data)

data = pad_sequences(sequences, maxlen=6)
print(data)

data = pad_sequences(sequences, maxlen=4) #RNN pays more attention to end of sentence
print(data)

len_seq = [len(i) for i in sequences].sort()
