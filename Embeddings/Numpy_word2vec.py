# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import matplotlib.pyplot as plt
import numpy as np
import operator
import torch
import nltk
import json
from nltk.corpus import brown
from datetime import datetime
from sklearn.utils import shuffle
device = torch.device('cpu')

nltk.download('brown')
corpus = brown

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])

def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
  sentences = brown.sents()
  indexed_sentences = []
  i = 0
  word2idx = {}
  idx2word = []
  word_idx_count = {}

  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        idx2word.append(token)
        word2idx[token] = i
        i += 1

      # keep track of counts for later sorting
      idx = word2idx[token]
      word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

      indexed_sentence.append(idx)
    indexed_sentences.append(indexed_sentence)

  # restrict vocab size
  # set all the words I want to keep to infinity
  # so that they are included when I pick the most common words
  for word in keep_words:
    word_idx_count[word2idx[word]] = float('inf')

  sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
  word2idx_small = {}
  new_idx = 0
  idx_new_idx_map = {}
  for idx, count in sorted_word_idx_count[:n_vocab]:
    word = idx2word[idx]
    word2idx_small[word] = new_idx
    idx_new_idx_map[idx] = new_idx
    new_idx += 1
  # let 'unknown' be the last token
  word2idx_small['UNKNOWN'] = new_idx
  unknown = new_idx

  for word in keep_words:
    assert(word in word2idx_small)

  # map old idx to new idx
  sentences_small = []
  for sentence in indexed_sentences:
    if len(sentence) > 1:
      new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
      sentences_small.append(new_sentence)

  return sentences_small, word2idx_small

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def init_weights(shape):
  return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))

class Model:
  def __init__(self, D, V, context_sz):
    self.D = D
    self.V = V
    self.context_sz = context_sz

  def _get_pnw(self, X): #Negative sampling probabilities
    word_freq = {}
    word_count = sum(len(x) for x in X)
    for x in X:
      for xj in x:
        if xj not in word_freq:
          word_freq[xj] = 0
        word_freq[xj] += 1
    self.Pnw = np.zeros(self.V)
    for j in range(2, self.V): #Exclude start and end tokens
      self.Pnw[j] = (word_freq[j] / float(word_count))**0.75 #Boost small prob words
    assert(np.all(self.Pnw[2:] > 0))
    return self.Pnw

  def _get_negative_samples(self, context, num_neg_samples):
    saved = {}
    for context_idx in context:
      saved[context_idx] = self.Pnw[context_idx] #we copy context to new dict, for memory efficiency
      #and restore them back later on.
      self.Pnw[context_idx] = 0
    neg_samples = np.random.choice(
        range(self.V),
        size=num_neg_samples,
        replace=False,
        p=self.Pnw / np.sum(self.Pnw)
    )
    for j, pnwj in saved.items():
      self.Pnw[j] = pnwj
    assert(np.all(self.Pnw[2:] > 0))
    return neg_samples

  def fit(self, X,epochs=10, num_neg_samples=10, lr=10e-5, mu=0.99, reg=0.1):
    N = len(X) #samples
    V = self.V #vocab size
    D = self.D #hidden dim/embedding
    self._get_pnw(X) #calculate negative probabilities for further sampling

    self.W1 = init_weights((V, D)) #encode embedding vector
    self.W2 = init_weights((D, V)) #decode embedding vector
    dW1 = np.zeros(self.W1.shape) #gradient W1
    dW2 = np.zeros(self.W2.shape) #gradient W2
    costs = []
    cost_per_epoch = []
    sample_indices = range(N)
    for i in range(epochs):
      t0 = datetime.now()
      sample_indices = shuffle(sample_indices)
      cost_per_epoch_i = []
      for it in range(N): #Loop of data generation (over sentences)
        j = sample_indices[it] #shuffle procedure
        x = X[j] #Current sentence

        if len(x) < 2 * self.context_sz + 1: #Sentence has to be at least
        #left_context + right_context + input long.
          continue

        cj = []
        n = len(x) #Sentence length
        for jj in range(n): #Loop of data generation (inside sentences)
          Z = self.W1[x[jj], :] #From input vector we take specific word representation
          start = max(0, jj - self.context_sz) #If current word is too close to
          #beginning of sentence, then we just take as much words as we can.
          end = min(n, jj + 1 + self.context_sz) #If context goes beyong sentence
          #length then we just take it up to the end of sentence.
          context = np.concatenate((x[start:jj], x[jj+1:end])) #All words that should be in the answer to input
          context = np.array(list(set(context)), dtype=np.int32) #Remove duplicates (Skipgram is CBOW after all)

          posA = Z.dot(self.W2[:,context]) #Multiply input vec (one word) 1xD by all output vecs (words in context) DxContext = 1xContext
          pos_pY = sigmoid(posA) #Prepare output for loss. We expect output to be as close to 1 as possible for all context words.
          #We punish model for any deviation from that.

          neg_samples = self._get_negative_samples(context, num_neg_samples) #Get neg. samples

          negA = Z.dot(self.W2[:, neg_samples]) #input vec on output vec generate 1xNeg_samples
          neg_pY = sigmoid(-negA) #Output for neg_samples is negative due to the target being reversed.
          #We need them to be close to 0 as much as we can.
          c = -np.log(pos_pY).sum() - np.log(neg_pY).sum()
          cj.append(c / (num_neg_samples + len(context))) #Save costs for model estimation

          pos_err = pos_pY - 1
          neg_err = 1 - neg_pY

          dW2[:,context] = mu*dW2[:,context] - lr*(np.outer(Z, pos_err) + reg*self.W2[:,context])
          dW2[:, neg_samples] = mu*dW2[:,neg_samples] - lr*(np.outer(Z,neg_err) + reg*self.W2[:,neg_samples])

          self.W2[:, context] += dW2[:, context]
          self.W2[:, neg_samples] += dW2[:, neg_samples]

          gradW1 = pos_err.dot(self.W2[:, context].T) + neg_err.dot(self.W2[:, neg_samples].T)
          dW1[x[jj], :] = mu*dW1[x[jj], :] - lr*(gradW1 + reg*self.W1[x[jj], :])

          self.W1[x[jj],:] += dW1[x[jj], :]

        cj = np.mean(cj)
        cost_per_epoch_i.append(cj)
        costs.append(cj)
        if it % 100 == 0:
          print(f"epoch: {i}, j: {it/N:.2f}, cost: {cj:.6f}")

      epoch_cost = np.mean(cost_per_epoch_i)
      cost_per_epoch.append(epoch_cost)
      print(f'time to complete epoch {i}: {datetime.now() - t0}, ""')

    plt.plot(costs)
    plt.title("Numpy costs")
    plt.savefig("Numpy_costs")
    plt.clf()

    plt.plot(cost_per_epoch)
    plt.title("Cost per epoch")
    plt.savefig("Cost_per_epoch")
    plt.clf()

  def save(self, fn):
    arrays = [self.W1, self.W2]
    np.savez(fn, *arrays)

def main():
  sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
  with open('w2v_word2idx.json', 'w') as f:
    json.dump(word2idx, f)

  V = len(word2idx)
  model = Model(80, V, 10)
  model.fit(sentences, epochs=20, lr=10e-6, mu=0.99, reg=0.1)
  model.save('w2v_model.npz')

def find_analogies(w1, w2, w3, concat=True, we_file='w2v_model.npz',  w2i_file='w2v_word2idx.npz'):
  npz = np.load(we_file)
  W1 = npz['arr_0']
  W2 = npz['arr_1']

  with open(w2i_file) as f:
    word2idx = json.load(f)

  V = len(word2idx)

  if concat: #Two ways of using word2vec. We can either find mean of W1 and W2 or concat them one after other.
  #We get matrix NxD if we use mean, or Nx2D if we use concate.
    We = np.hstack([W1, W2.T])
    print("We.shape", We.shape)
    assert(V==We.shape[0])
  else:
    We = (W1 + W2.T) / 2

if __name__ == '__main__':
  main()
  for concat in (True, False):
    print("*** concat:", concat)
    find_analogies('king', 'man', 'woman', concat=concat)
    find_analogies('france', 'paris', 'london', concat=concat)
    find_analogies('france', 'paris', 'rome', concat=concat)
    find_analogies('paris', 'france', 'italy', concat=concat)