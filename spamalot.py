from classifiers import classifier_menu, classifier_names, classifier_default
from tokenize import tokenize_menu, tokenize_names, tokenize_default
from feature import feature_menu, feature_names, feature_default
import sys
import math
import random
import collections
from optparse import OptionParser, OptionGroup

def main():
  global randgen
  options,args = parseCommandLine()

  if options.seed is None:
    seed = random.randrange(1000000)
    print("Random seed: " + str(seed))
  else:
    seed = options.seed

  randgen = random.Random(seed)
  raw = parseFile(options.filename)

  if options.downcase:
    raw = [ (sentence.lower(), label) for (sentence,label) in raw ]

  tokenize = tokenize_menu[options.tokenize][0]
  tokenized_data = [ (tokenize(sentence), label) for (sentence,label) in raw ]

  feature_alg = feature_menu[options.feature][0]
  dataset = [ (feature_alg(tokens), label) for (tokens,label) in tokenized_data ]

  classifier = classifier_menu[options.algorithm][0](options, randgen)
  overall_results = {}
  n = len(dataset)
  stripesize = n / 10.
  for t in range(options.rounds):
    trial_results = {}
    randgen.shuffle(dataset)
    for k in range(10):
      begin = int(round(k*stripesize))
      end = int(round((k+1)*stripesize))
      test = dataset[begin:end]
      training = dataset[0:begin] + dataset[end:]
      results = classifier.evaluate(training, test)
      
      if options.verbose:
        printResults("Stripe %d:" % (1+k), results)

      addResults(trial_results, results)

    addResults(overall_results, trial_results)
    if options.rounds > 1:
      printResults("Cross-validation Round %d:" % (1+t), trial_results)
          
  printResults('Overall:', overall_results)

def addResults(combined, extra):
  for (actual, secondary) in extra.items():
    combinedSecondary = combined.setdefault(actual,{})
    for (outcome,count) in secondary.items():
      combinedSecondary[outcome] = count + combinedSecondary.get(outcome, 0)

def printResults(preface, results):
  print(preface)
  correct = 0
  total = 0
  
  keys = results.keys()
  longest = max(keys, key=str.__len__)
  width = max(6, len(longest))
  print(' '*(2*width) + 'Predicted')
  header = ['Actual' + ' '*(width-5)]
  format = '  %%%ds' % width
  for actual in keys:
    header.append(format % actual)
  print ''.join(header)

  preface = '%%%ds:' % width
  format = '  %%%d.4f' % width
  for actual in keys:
    secondary = results[actual]
    subtotal = sum(secondary.values())
    total += subtotal
    correct += secondary.get(actual,0)
    line = [ preface % actual ]
    for outcome in keys:
      pct = 1.0 * secondary.get(outcome,0) / subtotal
      line.append(format % pct)
    print(''.join(line))

  print('correct classification: %-6.4f' % (1.0 * correct / total))
  print('')


def parseFile(filename):
  samples = []
  fp = open(filename)
  for line in fp:
    line = line.strip()
    if '\t' in line:
      pieces = line.split('\t')
      samples.append( (pieces[1], pieces[0]) )
  return samples

#------------------------------------------------------------------------
# Code for command line options
#------------------------------------------------------------------------
class MyParser(OptionParser):
  def format_epilog(self, formatter):
    return self.epilog  # without altering newlines

def parseCommandLine():
  epilog = ''
  epilog += ('\n  Classification Algorithms:\n' +
             '\n'.join(['    %-10s %s'%(name,classifier_menu[name][1]) for name in classifier_names]) + '\n')
  epilog += ('\n  Tokenizer Options:\n' +
             '\n'.join(['    %-10s %s'%(name,tokenize_menu[name][1]) for name in tokenize_names]) + '\n')
  epilog += ('\n  Feature Space Options:\n' +
             '\n'.join(['    %-10s %s'%(name,feature_menu[name][1]) for name in feature_names]) + '\n')

  
  parser = MyParser(usage='usage: %prog [options]', epilog=epilog)
  
  group = OptionGroup(parser, 'Experiment Options')
  group.add_option('-r', dest='rounds', type='int', default=1,
                   help='Number of independent rounds of 10-fold cross-validation [default: %default]')
  group.add_option('-i', dest='filename', default='SMS_Spam.dat',
                   help='read data from file [default: %default]')
  group.add_option('-s', dest='seed', type=int, default=None,
                   help='seed for all randomization [default: clock]')
  group.add_option('-v', dest='verbose', default=False, action='store_true',
                   help=('Verbose; print success rate for each independent trial [default: %default]'))
  parser.add_option_group(group)
  
  
  group = OptionGroup(parser, 'Algorithmic Options')
  group.add_option('-c', dest='algorithm', default = classifier_default, choices = classifier_names,
                   help=('Classification algorithm (see below) [default: %default]'))
  group.add_option('-t', dest='tokenize', default = tokenize_default, choices = tokenize_names,
                   help=('Tokenizer (see below) [default: %default]'))
  group.add_option('-d', dest='downcase', default=False, action='store_true',
                   help=('Downcase all messages before tokenizing [default: %default]'))
  group.add_option('-f', dest='feature', default = feature_default, choices = feature_names,
                   help=('Feature selection (see below) [default: %default]'))
  parser.add_option_group(group)

  group = OptionGroup(parser, 'Additional Nearest Neighbor Settings')
  group.add_option('-k', dest='knn', type=int, default=1, metavar="NEIGHBORS",
                   help=('Number of neighbors for k-nearest neighbors algorithm [default: %default]'))
  parser.add_option_group(group)

  group = OptionGroup(parser, 'Additional Naive Bayesian Settings')
  group.add_option('-b', dest='bayesStrength', type=float, default=0, metavar="STRENGTH",
                   help=('Strength factor of prior for Naive Bayesian [default: %default]'))
  parser.add_option_group(group)

  group = OptionGroup(parser, 'Additional Decision Tree Settings')
  group.add_option('-m', dest='treeThreshold', type=int, default=1, metavar="SIZE",
                   help=('Split decision subtrees having more than this many samples [default: %default]'))
  group.add_option('-u', dest='treeUniformity', type=float, default=1.0, metavar="PCT",
                   help=('Do not split decision subtree if uniformity is at or above given ratio [default: %default]'))
  parser.add_option_group(group)

  return parser.parse_args()
  

if __name__ == '__main__':
  print 'python ' + ' '.join(sys.argv)  # echo to capture configuration in standard out
  main()


