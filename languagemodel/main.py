from bigram import *

words = open("names.txt", "r").read().splitlines()

lm = Bigram(words)
print(lm.predict(15))
