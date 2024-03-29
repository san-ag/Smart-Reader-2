from collections import Counter
from textblob import TextBlob
import string
import math
import nltk
from nltk.util import ngrams
from nltk.corpus import brown
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
import os
from nltk.parse import stanford
import pexpect

PROJECT_HOME='/Users/sanchitagarwal/Desktop/11611-NLP/project'
PARSER_PATH=os.path.join(PROJECT_HOME, 'stanford-parser-full-2015-04-20')
PARSER_MODEL_PATH=os.path.join(PARSER_PATH, 'stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

os.environ['STANFORD_PARSER'] = PARSER_PATH
os.environ['STANFORD_MODELS'] = PARSER_PATH
os.environ['CLASSPATH'] = PARSER_PATH
parser = stanford.StanfordParser(model_path=PARSER_MODEL_PATH)

context_parse_map = {}

with open('data/stopList.txt','r') as f:
    stopWords = f.read().split()


class perplexityComputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lm1,lm2=None):
        self.lm1 = lm1
        self.lm2 = lm2
    
    def fit(self, X, y=None, **fit_params):
        return self

    def get_perpValue(self,check):
        perpValue = check.split('=')
        perpValue = perpValue[1].strip().split(',')[0]
        return float(perpValue)

    def transform(self, X):

        perps_lm1 = []
        perps_lm2 = []
        child_lm1 = pexpect.spawn(evallm_path+" -binary "+self.lm1)
        child_lm2 = pexpect.spawn(evallm_path+" -binary "+self.lm2)
        child_lm1.expect ('evallm :')
        child_lm2.expect ('evallm :')
        
        for question in X:
            
            #print question
            question = ' '.join(question.split()[0:-1]).upper()
            question = '<s> '+question+' <\s>'
            
            fpi = open("perp_input.txt","w")
            fpi.write(question)
            fpi.close()
            
            child_lm1.sendline("perplexity -text perp_input.txt")
            child_lm1.expect('evallm :')
            check = child_lm1.before
            perpValue = self.get_perpValue(check)
            perps_lm1.append(perpValue)
            
            child_lm2.sendline("perplexity -text perp_input.txt")
            child_lm2.expect('evallm :')
            check = child_lm2.before
            perpValue = self.get_perpValue(check)
            perps_lm2.append(perpValue)
        #close the programs
        child_lm1.sendline('quit')
        child_lm2.sendline('quit')
        a1 = np.array(perps_lm1)
        a2 = np.array(perps_lm2)
        
        #a1 = np.log2(a1)
        #a2 = np.log2(a2)
        
        afinal = np.column_stack((a1,a2))
        
        print afinal
        
        return afinal


def tag_POS(document):
    
    tagged=nltk.pos_tag(document.split(),tagset='universal')
    
    pos_tags = ''
    
    for item in tagged:
        #if item[0] == ('?'):
        #    continue
        #else:
        pos_tags+=item[1]+' '
    
    pos_tagged  = pos_tags.rstrip()
    
    return pos_tagged


def isPassiveVoice(question):
    
    isPassive = 0
    
    q = TextBlob(question)
    tags = q.tags
    
    toBeList = ["is","am","are","was","were","be","being"]
    
    for i in range(len(tags)-1):
        if tags[i][0] in toBeList and tags[i+1][1].startswith('V') and not tags[i+1][1].startswith('VBG'):
            isPassive = 1
    
    return isPassive

def get_wh_type(question):
    wh_type = 0
    
    wh_list = ['what','where','when','who','whom','whose',"which",'how'] 
    
    words = question.split()
    for idx,wh_word in enumerate(words):
    
    #wh_word = question.split()[0].lower()
        if wh_word in wh_list:
            wh_type = idx#wh_list.index(wh_word)
            break
    
    return wh_type


def isNegated(question):
    
    negation = 0
    
    negatives = ["no","never","not"]
    words = question.split()
    
    for neg in negatives:
        if neg in words:negation = 1
        
    return negation

def isPronounResolved(context):
    
    #parseTree = parser.raw_parse(context)
    #root = parseTree.next()
    
    root = context_parse_map[context]
    
    print root.pretty_print()
    
    np = root[0][0]
    if np.label() != "NP":
        return False
    flag = False
    for token in np:
        if token.label().startswith('PRP'):
            flag = False
            break
        if token.label().startswith('NN'):
            flag = True
    return flag

def countStopWords(question):
    
    stopCount = 0
    
    words = question.split()
    for word in words:
        if word in stopWords:stopCount+=1
        
    return stopCount

def countNamedEntities(question):
    count = 0
    
    tagged = nltk.pos_tag(question.split())
    for (token,tag) in tagged:
        if tag=='NNP':count+=1
    
    return count

def compute_overlap_score(c,q):
        
    c_set = set(c.split())
    q_set = set(q.split())
    
    common = c_set.intersection(q_set)
    
    score = float(len(common))
            
    return score

class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextPOSExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, context_question_list):
        features = np.recarray(shape=(len(context_question_list),),
                               dtype=[('question', object), ('question_pos', object),('context',object),('context_pos',object)])
        
        for i, (context,question) in enumerate(context_question_list):
            

            features['question'][i] = question

            features['question_pos'][i] = tag_POS(question)
            
            features['context'][i] = context

            features['context_pos'][i] = tag_POS(context)
            
            
        return features

class questionFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, questions):
                
        feature_dicts = []
        
        for q in questions:
            q = q.lower()
            f_d = {'voice':isPassiveVoice(q),'negation':isNegated(q),
                   'wh_type':get_wh_type(q),'q_len':1.0/len(q.split()),
                   'stopCount':countStopWords(q),'NECount':countNamedEntities(q)}
            feature_dicts.append(f_d)
            
        
        return feature_dicts

class contextFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, contexts):
                
        feature_dicts = []
        
        for c in contexts:
            #f_d = {'pronoun_resolution':isPronounResolved(c),'c_len':len(c.split())}
            f_d = {'c_len':1.0/len(c.split()),'NECount':countNamedEntities(c)}
            feature_dicts.append(f_d)
        
        
        return feature_dicts
    
class overlapFeature(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, dicts):
                
        scores = []
        
        contexts = dicts['context']
        questions = dicts['question']
        
        n = len(contexts)
        
        print n
        
        for i in range(n):
            context = contexts[i]
            question = questions[i]
            score = compute_overlap_score(context, question)
            scores.append(score)
            
            
        scores = np.array([scores])
        return scores.T
 

def constructPipeline():
    
    #lm_3gram_path = "language_models/LM-train-100MW-3gram.binlm"
    #lm_4gram_path = "language_models/LM-train-100MW-4gram.binlm"
    
    lm_3gram_path = "language_models/BROWN-2gram.binlm"
    lm_4gram_path = "language_models/BROWN-3gram.binlm"
    
    #lm_3gram_path = "language_models/BROWN_TREC-3-gram.binlm"
    #lm_4gram_path = "language_models/BROWN_TREC-4-gram.binlm"
    
    lm_2gram_pos_path = 'language_models/BROWN_TREC_POS-3gram.binlm'
    lm_3gram_pos_path = "language_models/BROWN_TREC_POS-4gram.binlm"
    

    
    
    pos_vectorizer = CountVectorizer(ngram_range=(2,3),tokenizer=lambda article:article.split(),
                                min_df=1,lowercase=False,max_features = 50,dtype = np.float64)
    
    normalizer = preprocessing.Normalizer(norm = 'l1')
    
    pipeline = Pipeline([
                        ('textPOS',TextPOSExtractor()),
                        ('features',FeatureUnion(
                                    transformer_list = [
                                            ('n_gram_pos_question', Pipeline([
                                                           ('selector', ItemSelector(key='question_pos')),
                                                            ('tf',pos_vectorizer),
                                                            ('normalize',normalizer),
                                                            ])),
                                                        
                                             #('n_gram_pos_context', Pipeline([
                                             #              ('selector', ItemSelector(key='context_pos')),
                                             #               ('tf',pos_vectorizer),
                                             #               ('normalize',normalizer),
                                             #               ])),
                    
                                            ('question_based',Pipeline([
                                                            ('selector', ItemSelector(key='question')),
                                                            ('q_based',questionFeatures()),
                                                            ('vect', DictVectorizer()),
                                                            ('normalize',normalizer),
                                                            ])), 
                                            #('context_based',Pipeline([
                                            #                ('selector', ItemSelector(key='context')),
                                            #                ('count-based',contextFeatures()),
                                            #                ('vect', DictVectorizer()),
                                            #                ('normalize',normalizer),
                                            #                ])), 
                                            ('perplexity',Pipeline([
                                                            ('selector', ItemSelector(key='question')),
                                                            ('per',perplexityComputer(lm_3gram_path,lm_4gram_path)),
                                                            ('normalize',normalizer)
                                                            ])),
                                            #('perplexity',Pipeline([
                                            #                ('selector', ItemSelector(key='question_pos')),
                                            #                ('per',perplexityComputer(lm_2gram_pos_path,lm_3gram_pos_path)),
                                            #                ('normalize',normalizer)
                                            #               ])),
                                            ('overlap',Pipeline([
                                                            ('ovr',overlapFeature()),
                                                            ('normalize',normalizer)
                                                            ])),
                                                        ],
                                    transformer_weights={
                                        'n_gram_pos_question': 1,
                                        'perplexity': 1,
                                        'question_based':0.8,
                                        'overlap':0.5
                                    },             
                                )),
                        ])

    return pipeline,pos_vectorizer

def generate_context_parseTrees(context_question_list):
    
    global context_parse_map
    
    for item in context_question_list:
        context = item[1]
        if context not in context_parse_map:
            parseTree = parser.raw_parse(context)
            root = parseTree.next()
            context_parse_map[context] = root
    
    print context_parse_map

#fix it for train and test
def extractFeatures(context_question_list):
    
    
    #construct a map from context to root of its parseTree
    
    #generate_context_parseTrees(context_question_list)
    
    global evallm_path
    evallm_path = '/Users/sanchitagarwal/Desktop/Directed-Study-II/QuestionRanker/Ranker/CMU-Cam_Toolkit_v2/bin/evallm'

    data = []
    
    for item in context_question_list:
        context = item[1]
        question = item[2]
        data.append(tuple([context,question]))


    pipeline,pv = constructPipeline()
    
    
    
    print pipeline
    #print pipeline.get_feature_names()
    featureMatrix = pipeline.fit_transform(data)
    print featureMatrix.shape
    
    #print pv.get_feature_names()

    return featureMatrix
