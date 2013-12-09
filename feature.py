#------------------------------------------------------------------------
# Menu of feature generation functions
#
# A general feature space function should accept an ordered list of token strings,
# and return a dictionary that maps feature names to feature values.
#------------------------------------------------------------------------

def feature_token_multiset(tokens):
  """Returns map from tokens to number of occurrences."""
  return collections.Counter(tokens)

def feature_token_set(tokens):
  """Returns map from each token to the constant 1."""
  return { token : 1 for token in tokens }

# map from short label to (function,description) tuple
feature_menu = {
  'exists' : (feature_token_set, "Use zero-one variable for which tokens occur"),
  'freq' : (feature_token_multiset, "Map token to its frequency in the message"),
  # more may be added here...
  }

feature_names = tuple(sorted(feature_menu.keys()))
feature_default = 'exists'
