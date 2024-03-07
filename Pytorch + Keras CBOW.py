# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:24:41 2024

@author: Bannikov Maxim
"""

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
dataset = api.load("text8")

i = 0
for x in dataset:
    print(x)
    i += 1
    if i > 10:
        break
    
i = 0
for x in dataset:
    i += 1
print(i)

doc_lengths = []
for x in dataset:
    l = len(x)
    doc_lengths.append(l)

np.mean(doc_lengths), np.std(doc_lengths)

from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 20000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(dataset)
sequences = tokenizer.texts_to_sequences(dataset)
data_length = len(sequences)

type(sequences)

tokenizer.num_words
len(tokenizer.word_index)

tokenizer.index_word

context_size = 10
embedding_dim = 50
batch_size = 128
lr = 10e-3
num_epochs = 10000
losses = []
acc_list = []

class CBOW(nn.Module):
    def __init__(self, embedding_dim, vocab_size=-1):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        out = self.embeddings(x).mean(1).squeeze(1)
        out = self.l1(out)
        return out

model = CBOW(embedding_dim, vocab_size)

class MyDataset(torch.utils.data.Dataset):

  def __init__(self, sequences, context_size):
        self.sequences = sequences
        self.context_size = context_size
        self.half_context_size = context_size // 2
        
  def __len__(self):
        return len(self.sequences)

  def __getitem__(self, idx):
        x = np.zeros(self.context_size)
        seq = sequences[idx]
        j = np.random.randint(0, len(seq) - self.context_size - 1)
        x1 = seq[j:j + self.half_context_size]
        x2 = seq[j + self.half_context_size + 1:j + self.context_size + 1]
        x[:self.half_context_size] = x1
        x[self.half_context_size:] = x2
        X = torch.tensor(x, dtype=torch.int64)
        y = seq[j + self.half_context_size]
        Y = torch.tensor(y, dtype=torch.int64)
        return X, Y

data = MyDataset(sequences, context_size)
train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
n_total_steps = len(train_loader)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
    
        x = x.to(device)
        y = y.to(device)
      
        #forward
        outputs = model(x)
        loss = criterion(outputs, y)
      
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            losses.append(loss.item())
            _, predictions = torch.max(outputs, 1)
            n_samples = y.shape[0]
            n_correct = (predictions == y).sum().item()
            train_acc = 100.0 * n_correct / n_samples
            acc_list.append(train_acc)
    
plt.plot(losses)
plt.yscale('log')
plt.show()

plt.plot(acc_list)
plt.yscale('log')
plt.show()

embeddings = model.embeddings.weight.detach().numpy()
    
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
neighbors.fit(embeddings)

#KNN for word

def print_neighbors(query):
    query_idx = tokenizer.word_index[query]
    query = embeddings[query_idx:query_idx + 1]
    distances, indices = neighbors.kneighbors(query)

    for idx in indices[0]:
        word = tokenizer.index_word[idx]
        print(word)
        
print_neighbors('queen')
print_neighbors('uncle')
print_neighbors('paris')
print_neighbors('japan')

#KNN for word

def get_embedding(word):
    idx = tokenizer.word_index[word]
    return embeddings[idx:idx + 1]

king = get_embedding('king')
man = get_embedding('man')
woman = get_embedding('woman')
query = king - man + woman

distances, indices = neighbors.kneighbors(query)
for idx in indices[0]:
    word = tokenizer.index_word[idx]
    print(word)

england = get_embedding('england')
english = get_embedding('english')
australian = get_embedding('australian')
query = england - english + australian

distances, indices = neighbors.kneighbors(query)
for idx in indices[0]:
    word = tokenizer.index_word[idx]
    print(word)
    
    