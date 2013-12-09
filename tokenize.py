#------------------------------------------------------------------------
# Menu of tokenizers
#
# A tokenizer is a function that takes a raw string an returns an ordered
# list of tokens.
#------------------------------------------------------------------------

def tokenize_whitespace(message):
  return message.split()


# map from short label to (function,description) tuple
tokenize_menu = {
  'default' : (tokenize_whitespace, "Use original whitespace-separated tokens"),
#  'alpha' : (tokenize_alpha, "Break on any nonalpha (other than apostrophe)")
  }

tokenize_names = tuple(sorted(tokenize_menu.keys()))
tokenize_default = 'default'
