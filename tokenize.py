#------------------------------------------------------------------------
# Menu of tokenizers
#
# A tokenizer is a function that takes a raw string an returns an ordered
# list of tokens.
#------------------------------------------------------------------------
import re
def tokenize_whitespace(message):
    return message.split()

#tokenizer to that removes all puncation expect apostrophes
def tokenize_alpha(message):
    #tokenizer to do better than default takes into account punctuation
    wordList = re.sub(r"[^\w']"," ",  message).split()
    return wordList

# tokenizer that removes all puncation and apostrophes
def tokenize_alpha2(message):
    wordList = re.sub(r"[^\w]"," ",message).split()
    return wordList

#tokenizer that removes all puncation expect "I'm"
def tokenize_alpha3(message):
    wordList= re.sub("[^\w]"," ",message).split()
    count = 0
    for tag in wordList:
        if tag is "I" or tag is "i":
            if count+1 == len(wordList):
                return wordList
            elif  wordList[count+1] is "M" or wordList[count+1] is "m":
                wordList.append("I'm")
                wordList.pop(count)
                if count+1 <= len(wordList):
                    wordList.pop(count+1)
        count += 1
    return  wordList

# tokenizer that turnes upper case letters to lower and removes puncation
def tokenize_alpha4(message):
    newmessage= message.lower()
    wordList= re.sub(r"[\w']"," ",newmessage).split()
    return wordList



# map from short label to (function,description) tuple
tokenize_menu = {
    'default' : (tokenize_whitespace, "Use original whitespace-separated tokens"),
    'alpha' : (tokenize_alpha, "Break on any nonalpha (other than apostrophe)"),
    'alpha2': (tokenize_alpha2, "Breaks on any nonalpha characters"),
    'alpha3': (tokenize_alpha3, "Breaks on any nonalpha characters expect i'm"),
    'alpha4': (tokenize_alpha4, "makes strign lower case breaks on any nonalpha character")
}


tokenize_names = tuple(sorted(tokenize_menu.keys()))
tokenize_default = 'default'
tokenize_alpha= 'alpha'
tokenize_alpha2= 'alpha2'
tokenize_alpha3= 'alpha3'
tokenize_alpha4= 'alpha4'

