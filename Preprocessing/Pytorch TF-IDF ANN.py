# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import requests
import io
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_hidden = 300
num_classes = 10
num_epochs = 5
batch_size = 50
lr = 0.001

response = requests.get('https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv')
file_object = io.StringIO(response.content.decode('utf-8'))
df = pd.read_csv(file_object)

stops = set(stopwords.words('english'))
stops = stops.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two', 'mr', 'new', 'year'
})


vectorizer = TfidfVectorizer(stop_words=list(stops))
X = vectorizer.fit_transform(df['text']).todense()
num_features = X.shape[1]
lb = LabelEncoder()
Y = list(lb.fit_transform(df['labels']))
num_classes = max(Y) + 1

class MyDataset(Dataset):
  def __init__(self, X, Y):
    self.x=torch.tensor(X, dtype=torch.float32)
    self.y=torch.tensor(Y, dtype=torch.int64)

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

data = MyDataset(X, Y)
train_indices, test_indices, _, _ = train_test_split(data.x, data.y, test_size=0.2)

train_dataset, test_dataset = random_split(data, [0.8, 0.2])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(num_features, num_hidden)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out

model = NeuralNet(num_features, num_hidden, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)
losses = []
acc_list = []

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

#test
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)

    #value, index
    _, predictions = torch.max(outputs, 1)
    n_samples += y.shape[0]
    n_correct += (predictions == y).sum().item()
  acc = 100.0 * n_correct / n_samples
  print(f'accuracy = {acc}')

plt.plot(losses, label = 'train loss')
plt.legend()

plt.plot(acc_list, label = 'train acc')
plt.legend()
