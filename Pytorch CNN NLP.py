# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:12:19 2024

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import sklearn
import io
import torch
import torch.nn as nn
import torch.nn.functional as f
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from string import digits
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


device = torch.device('cpu') 
nltk.download('stopwords')
stops = stopwords.words('english')
punkt = string.punctuation

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(file_object, dtype={'text': 'string', 'labels': 'string'})
df.text = df.text.str.lower()
df.text = df.text.str.strip()
remove_digits = str.maketrans('', '', digits)
df.text = df.text.str.translate(remove_digits)
remove_punkt = str.maketrans('', '', punkt)
df.text = df.text.str.translate(remove_punkt)
df.text = df.text.apply(lambda seq: ' '.join(word.lower() for word in seq.split() 
                                             if word not in stops
                                             and len(word) > 2))
df["targets"] = df.labels.astype('category').cat.codes
categories = len(set(df["targets"]))

df_train, df_test = train_test_split(df, test_size = 0.3)

vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df_train.text)
sequences_train = tokenizer.texts_to_sequences(df_train.text)
sequences_test = tokenizer.texts_to_sequences(df_test.text)

data_train = pad_sequences(sequences_train)
T = data_train.shape[1]
data_test = pad_sequences(sequences_test, maxlen = T)
cat_train = np.array(df_train.targets, dtype = 'int64')
cat_test = np.array(df_test.targets, dtype = 'int64')

embedding_dim = 50

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, categories):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, categories)
    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = f.max_pool1d(x, kernel_size=x.size(2)).squeeze()
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.int64)
        self.y = torch.tensor(y, dtype=torch.int64)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


batch_size = 10
train_dataset = CustomDataset(data_train, cat_train)
test_dataset = CustomDataset(data_test, cat_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
n_train_steps = len(train_loader)
n_test_steps = len(test_loader)

train_epoch_loss = []
test_epoch_loss = []
train_epoch_acc = []
test_epoch_acc = []

num_epochs = 50
lr = 10e-5
weight_decay = 10e-6

model = CNN(vocab_size, embedding_dim, categories)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#training loop
for epoch in range(num_epochs):
    losses = []
    acc_list = []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prediction = torch.max(output, 1).indices
        losses.append(loss.item())           
        n_samples = y.shape[0]
        n_correct = (prediction == y).sum().item()
        acc = 100.0 * n_correct / n_samples
        acc_list.append(acc)

    train_epoch_loss.append(sum(losses)/n_train_steps)
    train_epoch_acc.append(sum(acc_list)/n_train_steps)
    
    with torch.no_grad():
        losses = []
        acc_list = []
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            prediction = torch.max(output, 1).indices
            losses.append(loss.item())
            n_samples = y.shape[0]
            n_correct = (prediction == y).sum().item()
            acc = 100.0 * n_correct / n_samples
            acc_list.append(acc)
            
        test_epoch_loss.append(sum(losses)/n_test_steps)
        test_epoch_acc.append(sum(acc_list)/n_test_steps)
    
    print(f'epoch {epoch+1} / {num_epochs}, loss = {train_epoch_loss[-1]:.4f}, accuracy = {train_epoch_acc[-1]:.2f}')
    print(f'test loss = {test_epoch_loss[-1]:.4f}, test accuracy = {test_epoch_acc[-1]:.2f}')

plt.plot(train_epoch_loss, label='train_loss')
plt.plot(test_epoch_loss, label='test_loss')
plt.legend()
plt.show()

plt.plot(train_epoch_acc, label='train_acc')
plt.plot(test_epoch_acc, label='test_acc')
plt.legend()
plt.show()

with torch.no_grad():
    predictions = []
    outputs = []
    targets = []
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        prediction = torch.max(output, 1).indices
        predictions.append(prediction)
        outputs.append(f.softmax(output, dim=1))
        targets.append(y)
        
    predictions = np.concatenate(predictions)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

f1 = sklearn.metrics.f1_score(targets, predictions, average = 'weighted')
roc_auc = sklearn.metrics.roc_auc_score(targets, outputs, average = 'weighted', multi_class = 'ovr')
gini = 2 * roc_auc - 1

print(f'f1 score: {f1:.4f}, roc-auc score: {roc_auc:.4f}, gini score: {gini:.4f}')