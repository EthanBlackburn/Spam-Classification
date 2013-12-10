Our test results show the following for each classifier
KNN - ~90% accurate
Naive Bayes - ~95% accurate
Decision Tree - ~92% accurate
Modified Naive Bayes - I added weights to each word when counting them as spam or ham.
So, for each word in a spam message, the traditional bayes simply increments the words
by 1, whereas I incremented by 1.0/len(message). The idea behind this was that messages
such as "Free cable" would give higher spamicity to the words "free" and "cable" than
the message, "Do you know where I can get free cable?" would to each of its words. I 
did this in hopes that words that are likely indications of spam, such as "free" and
"mortgage", give the classifier a larger pull towards spam. After testing, it seemed to
only improve the results by a few tenths of a percent.