from __future__ import division
from abc import ABCMeta, abstractmethod
from collections import Counter
from math import log
import math
import heapq
import random
import operator
from sets import Set

#----------------------------------------------------------------------------
# Abstract base class for generic classifier
#
# This provides support for the evaluation experiment.
#----------------------------------------------------------------------------
class Classifier:
  """An abstract class representing a generic classifier algorithm."""
  __metaclass__ = ABCMeta

  def __init__(self, options, randgen):
    self.options = options
    self.random = randgen

  @abstractmethod
  def train(self, examples):
    """Performs training given a training set.

    Examples will be a list of (sample, label) pairs, with each sample
    being a dictionary mapping feature names to feature values. For convenience,
    it may be that certain samples are "missing" a feature, which we view as
    having an implicit None value.
    """

  @abstractmethod
  def classify(self, sample):
    """Return the classification for a single instance of the domain."""

  def evaluate(self, training_samples, test_samples):
    """Perform an evaluation of the algorithm.

    This algorithm returns a dictionary of dictionaries, so that
    result[actual][predicted] is the number of times that a sample
    with the given actual label is given the predicted label.
    """
    self.train(training_samples)
    result = {}
    for (features, answer) in test_samples:
      predicted = self.classify(features)
      secondary = result.setdefault(answer,{})
      secondary[predicted] = 1 + secondary.get(predicted, 0)
    return result

#----------------------------------------------------------------------------
# Greedy classifier
#----------------------------------------------------------------------------
class GreedyClassifier(Classifier):
  """
  This classifier assigns all samples to whatever label was the most common in the
  training set.
  """
  def train(self, examples):
    counter = Counter()
    for sample in examples:
      tag = sample[1]
      counter[tag] += 1
    self.choice = counter.most_common(1)[0][0]

  def classify(self, sample):
    return self.choice

#----------------------------------------------------------------------------
# Random classifier
#----------------------------------------------------------------------------
class RandomClassifier(Classifier):
  """
  This classifier randomly categorizes samples based solely
  on the a priori percentages from the training set.

  For binary classification, expected performance will be
  p*p + (1-p)*(1-p).
  """
  def train(self, examples):
    self.counts = {}
    self.total = len(examples)
    for sample in examples:
      tag = sample[1]
      self.counts[tag] = 1 + self.counts.get(tag, 0)

  def classify(self, sample):
    k = self.random.randrange(self.total)
    for (tag,count) in self.counts.items():
      if k < count:
        return tag
      else:
        k -= count

#----------------------------------------------------------------------------
# k-Nearest Neighbor classifier
#----------------------------------------------------------------------------
class NearestNeighborClassifier(Classifier):
  """
  This classifier assigns a sample to the majority class of its k-nearest neighbors
  (flipping a coin in case of a tie).
  """

  def train(self, examples):
    self.messages = {}
    self.words = Set()
    for message in examples:
      for word in message[0]:
        self.words.add(word)
    for i in range(len(examples)):
      self.messages[i] = {}
      self.messages[i][0] = Set()
      self.messages[i][0].update(examples[i][0])
      self.messages[i][1] = examples[i][1]
	  
  def classify(self, sample):
    neighbors = []
    min = len(self.words)
    self.sampleV = Set()
    Rank = 0
    self.sampleV.update(sample)
    for i in range(len(self.messages)):
      if(len(self.sampleV.symmetric_difference(self.messages[i][0])) < min):
        if(len(neighbors) >= self.options.knn):
          neighbors.pop()
        neighbors.append(self.messages[i])
        neighbors.sort(key=lambda x:len(self.sampleV.symmetric_difference(x[0])))
        min = len(self.sampleV.symmetric_difference(neighbors[-1][0]))
        
    for n in neighbors:
      if n[1] == "ham":
        Rank = Rank + 1
      else:
        Rank = Rank - 1
    if(Rank < 0):
      return "spam"
    elif(Rank >0):
      return "ham"
    else:
      return random.sample(["ham","spam"],1)[0] 
    		



#----------------------------------------------------------------------------
# Naive Bayesian classifier
#----------------------------------------------------------------------------
class NaiveBayesianClassifier(Classifier):
  """
  A naive Bayesian classifier.

  For quality results, callers should ensure that likely noise features are removed.
  """

  def train(self, examples):
    self.classified = Counter()
    self.hamWords = Counter()
    self.spamWords = Counter()
    for message in examples:
      self.classified[message[1]] += 1.0
      if(message[1] == "ham"):
        for word in message[0]:
          self.hamWords[word] += 1.0
      else:
        for word in message[0]:
          self.spamWords[word] += 1.0

  def classify(self, sample):
    Pspam = 1.0
    Pham = 1.0
    PS = float(self.classified["spam"])/(self.classified["spam"]+self.classified["ham"])
    PH = float(self.classified["ham"])/(self.classified["spam"]+self.classified["ham"])
    for word in sample:
      if(self.hamWords[word] or self.spamWords[word]):
        PWS = (self.spamWords[word]/self.classified["spam"])*PS
        PWH = (self.hamWords[word]/self.classified["ham"])*PH
        PSW = PWS/(PWS+PWH)
        PHW = PWH/(PWS+PWH)
        k = self.hamWords[word]+self.spamWords[word]
        b = self.options.bayesStrength
        Pspam *= (PS*b + k*PSW)/(b+k)
        Pham *= (PH*b + k*PHW)/(b+k)
    if(Pspam > Pham):
      return "spam"
    else:
      return "ham"
      
  
#----------------------------------------------------------------------------
# Decision Tree classifier
#----------------------------------------------------------------------------
class DecisionTreeClassifier(Classifier):
  """
  A classifier based on a decision tree.
  """
  def train(self, examples):
    messages = []
    classified = Counter()
    words = Set()
    self.tree = {}
    for message in examples:
      messages.append(message)
      for word in message[0]:
        words.add(word)
    self.subtree(messages,words,0)
    print self.tree
    
      
  def subtree(self, messages,words, depth):
    w = words
    m = messages
    d = depth
    if(len(m) <= self.options.treeThreshold):
      c = 0
      for mess in m:
        if mess[1] == "ham":
          c += 1
        else:
          c -= 1
      if(c >= 0):
        cl = "ham"
      else:
        cl = "spam"
      self.tree[depth] = cl
      return 
    temp = Counter()
    for message in m:
      temp[message[1]] +=1
    if(float(temp["ham"])/len(m) >= self.options.treeUniformity):
      self.tree[depth] = "ham"
      return
    elif(float(temp["spam"])/len(m) >= self.options.treeUniformity):
      self.tree[depth] = "spam"
      return
    entropy = (1000000,"arbitrary")
    classified = Counter()
    for word in w:
      for mess in m:
        if(word in mess[0]):
          classified[word] += 1.0
    for word in classified:
      if(classified[word] == 1.0):
        w.remove(word)
    for word in classified:
      if(classified[word] > 1.0):
        y = classified[word]/len(m)
        n = float(len(m)-classified[word])/len(m)
        e = float(-y*log(y) - n*log(n))
        if(e < entropy[0]):
          entropy = (e,word)
    self.tree[depth] = entropy[1]
    noWord = []
    yesWord = []
    if(not(entropy[1] == "arbitrary")):
      for mess in m:
        if entropy[1] in mess[0]:
          yesWord.append(mess)
        else:
          noWord.append(mess)
      w.remove(entropy[1])
    self.subtree(yesWord,w,2*d+2)
    self.subtree(noWord,w,2*d+1)
          
  def classify(self, sample):
    return self.traverse(sample,0)
      
  def traverse(self, message, depth):
    if(self.tree[depth] == "ham"):
      return "ham"
    elif(self.tree[depth] == "spam"):
      return "spam"
    elif(self.tree[depth] in message):
      return self.traverse(message,2*depth+2)
    else:
      return self.traverse(message,2*depth+1)

#----------------------------------------------------------------------------
# Modified Naive Bayesian classifier
#----------------------------------------------------------------------------
class ModifiedNaiveBayesianClassifier(Classifier):
  """
  A naive Bayesian classifier.

  For quality results, callers should ensure that likely noise features are removed.
  """

  def train(self, examples):
    self.classified = Counter()
    self.hamWords = Counter()
    self.spamWords = Counter()
    for message in examples:
      self.classified[message[1]] += 1.0
      if(message[1] == "ham"):
        for word in message[0]:
          self.hamWords[word] += 1.0/len(message[0]) 
      else:
        for word in message[0]:
          self.spamWords[word] += 1.0/len(message[0])

  def classify(self, sample):
    Pspam = 1.0
    Pham = 1.0
    PS = float(self.classified["spam"])/(self.classified["spam"]+self.classified["ham"])
    PH = float(self.classified["ham"])/(self.classified["spam"]+self.classified["ham"])
    for word in sample:
      if(self.hamWords[word] or self.spamWords[word]):
        PWS = (self.spamWords[word]/self.classified["spam"])*PS
        PWH = (self.hamWords[word]/self.classified["ham"])*PH
        PSW = PWS/(PWS+PWH)
        PHW = PWH/(PWS+PWH)
        k = self.hamWords[word]+self.spamWords[word]
        b = self.options.bayesStrength
        Pspam *= (PS*b + k*PSW)/(b+k)
        Pham *= (PH*b + k*PHW)/(b+k)
    if(Pspam > Pham):
      return "spam"
    else:
      return "ham"
#------------------------------------------------------------------------
# Menu of classifier algorithms
#------------------------------------------------------------------------
classifier_menu = {
  'greedy' : (GreedyClassifier, "Classify each sample to whatever outcome was most likely in the training"),
  'random' : (RandomClassifier, "Classify each sample randomly according to overall distribution of outcomes"),
  'neighbor' : (NearestNeighborClassifier, "k-Nearest Neighbor classifier"),
  'bayes' : (NaiveBayesianClassifier, "Naive Bayesian classifier"),
  'tree' : (DecisionTreeClassifier, "Decision tree classifier"),
  'modifiedBayes' : (ModifiedNaiveBayesianClassifier, " Modified Naive Bayesian classifier"),
  }

classifier_names = tuple(sorted(classifier_menu.keys()))
classifier_default = 'greedy'

