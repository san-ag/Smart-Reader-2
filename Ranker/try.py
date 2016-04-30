from collections import Counter
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer


def sklearn_try():
    
    corpus = ['How many gunners who died should there be a fitting public memorial to?','What boosted their capabilities?']
    
    doc_vectorizer = CountVectorizer(ngram_range=(1,1),tokenizer=lambda article:article.split(' '),
                                min_df=1,lowercase=False,max_features = 20000)
    
    X = doc_vectorizer.fit_transform(corpus)
    
    print X
    
    rows,columns = X.shape
    
    for vector in range(rows):
        for feature in range(columns):
            print str(X[vector,feature])+" "
        print ''
            

def ngram(question):
    
    tagged=nltk.pos_tag(question.split(' '))
    print tagged
    
    pos_tags = ''
    
    for item in tagged:
        if item[0] == ('?'):
            continue
        elif item[1].startswith('NN'):
            pos_tags+='NN '
        elif item[1].startswith('VB'):
            pos_tags+='VB '
        else:
            pos_tags+=item[1]+' '
    
    pos_tagged  = pos_tags.rstrip()
    
    print pos_tagged
    
    pos_unigrams = Counter(pos_tagged.split())
    pos_bigrams = Counter(ngrams(pos_tagged.split(),2))
    
    print pos_unigrams
    print pos_bigrams 

def tag_POS(document):
    
    tagged=nltk.pos_tag(document.split(' '),tagset='universal')
    
    print tagged
    
    pos_tags = ''
    
    for item in tagged:
        if item[0] == ('?'):
            continue
        else:
            pos_tags+=item[1]+' '
    
    pos_tagged  = pos_tags.rstrip()
    
    return pos_tags


question = 'How many gunners who died should there be a fitting public memorial to ?'
#ngram(question)
#sklearn_try()

print tag_POS(question)