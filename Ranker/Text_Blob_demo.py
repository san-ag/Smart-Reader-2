#testing various functions in textBlob library.

from collections import Counter 

from textblob import TextBlob
from textblob.taggers import NLTKTagger
from textblob import Word
from textblob.classifiers import NaiveBayesClassifier
from itertools import dropwhile

wiki = TextBlob("Python is a high-level, general-purpose programming language.")

print wiki.tags
#print wiki.noun_phrases
print Counter([pair[1] for pair in wiki.tags])

#print wiki.words

for pair in wiki.tags:
    if pair[1].find('NN')!=-1:print pair[0]
    
sentence = "Delhi is the capital of India. I love India"

t = TextBlob(sentence,pos_tagger=NLTKTagger())

print t.tags

word = Word('python')

print word.definitions

print t.translate(to="ko")


train = [
         ('I love this sandwich.', 'pos'),
         ('this is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('this is my best work.', 'pos'),
         ("what an awesome view", 'pos'),
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('he is my sworn enemy!', 'neg'),
         ('my boss is horrible.', 'neg')
         ]

cl = NaiveBayesClassifier(train)

prob_dist = cl.prob_classify("How you doing")

cl.show_informative_features(5) 

txt_A = TextBlob("He can climb the mountain")
txt_B = TextBlob("The mountain can be climbed by him")
txt_C = TextBlob("He is doing his homework")
txt_D = TextBlob("The homework is being done by him")

print txt_A.tags
print txt_B.tags
print txt_C.tags
print txt_D.tags



def passivep(tags):
    """Takes a list of tags, returns true if we think this is a passive
    sentence."""
    # Particularly, if we see a "BE" verb followed by some other, non-BE
    # verb, except for a gerund, we deem the sentence to be passive.
    
    postToBe = list(dropwhile(lambda(tag): not tag.startswith("BE"), tags))
    nongerund = lambda(tag): tag.startswith("V") and not tag.startswith("VBG")

    filtered = filter(nongerund, postToBe)
    out = any(filtered)

    return out

tagged = txt_B.tags
tags = map( lambda(tup): tup[1], tagged)
print passivep(tags)