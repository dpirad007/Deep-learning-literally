from bigram import *

words = open("names.txt", "r").read().splitlines()

lm = Bigram(words)
lm.predict(15)
