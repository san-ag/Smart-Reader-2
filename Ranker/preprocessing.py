#@author: sagarwa1@cs.cmu.edu
#date: 04/10/2016

#This code extracts the questions, text (sentence) from which the question 
#was generated and ratings from the QGSTEC-2010 test data. The extracted 
#data is written to gs.txt which serves as the gold standard.

#The data-set is taken from here: https://github.com/bjwyse/QGSTEC2010 

#There are in total 896 questions in the data-set that have been generated
#from 90 sentences. Each question has been rated by 2 raters on 5 
#criteria - questionType, relevance, correctness, ambiguity and variety.

#I need a generic score for each question, therefore, I chose to keep three
#of these scores - correctness, relevance and ambiguity. For each category,
#I take the average of the ratings by the two raters. Finally, to convert
#to a single score, I take the average score of the three categories.

#Each line in gs.txt is of the form
#id \t text \t question \t score

from nltk.corpus import brown
import nltk
import xml.etree.ElementTree as ET
import codecs
import random
import re
from _collections import defaultdict
from collections import Counter
import operator


def get_single_score(rater1,rater2):
    
    ambiguity = (float(rater1['ambiguity'])+float(rater2['ambiguity']))/2;
    relevance = (float(rater1['relevance'])+float(rater2['relevance']))/2;
    correctness = (float(rater1['correctness'])+float(rater2['correctness']))/2;
    
    #score = (ambiguity*3+relevance*4+correctness*4)/(3+4+4)
    score = correctness
    
    return score

def extract_data(input_file,output_file):
    
    lines = []
    
    tree = ET.parse(input_file)
    root = tree.getroot()

    
    for child in root:
        line = ''
        
        id = child.attrib['id']

        for node in child:
  
            if node.tag =='text':
                text = node.text
            if node.tag == 'submission':
                for q in node:
                    question = q.text.encode('ascii','ignore').strip()[0:-1]+' ?'
                    rater1 = q[0].attrib
                    rater2 = q[1].attrib
                    score = get_single_score(rater1, rater2)
                    
                    line = id+'\t'+text+'\t'+question+'\t'+str(score)
                    lines.append(line)
                    
    
    with codecs.open(output_file,'w',encoding='ascii',errors='ignore') as f:
        for line in lines:
            f.write('%s\n'%line)
            

#collapses duplicate questions from the gold standard.
def clean_gs(gs,gs_cleaned):
    
    uniqueQuestions = defaultdict(str)
    
    with open(gs,'r') as f:
        for line in f:
            q = line.split('\t')[2]
            uniqueQuestions[q] = line
            
    print len(uniqueQuestions)
    
    uniqueQuestionsSorted = sorted(uniqueQuestions.items(), key=lambda (k,v): int(v.split('\t')[0]))
    
    
    with open(gs_cleaned,'w') as f:
        for (k,v) in uniqueQuestionsSorted:
            f.write(v)
        
            
def prepareHeilmanQuestions(qFile,test_file):
    
    prev_id = '0'
    
    content = []
    
    with open(qFile,'r') as f:
        for line in f:
            parts = line.split('\t')
            content.append(parts)
            
    with open(test_file,'w') as f:
        
        for line in content:
            qid = line[0]
            ques = line[1]
            text = line[2]
            
            if qid!=prev_id:
                f.write('\n\n'+qid+'\t'+text+'\n\n')

            f.write(ques+'\n')
            prev_id = qid

#generates a corpus of pos-tags using brown corpus
def prepare_tagged_corpus():
    
    #V = Counter()
    
    tagged_list = []
    
    tagged_sentences = brown.tagged_sents(tagset='universal')
    
    for sent in tagged_sentences:
        tagged = [x[1] for x in sent]
        #V.update(tagged)
        sentence = ' '.join(tagged)
        sentence = '<s> '+sentence+'</s>'
        sentence = re.sub('\.+','',sentence)
        sentence = re.sub(' +',' ',sentence)
        tagged_list.append(sentence)
        
    
    with open('data/brown_POS.txt','w') as f:
        for item in tagged_list:
            f.write(item+'\n')
        
    #print V

#cleans and saves the brown corpus on disk       
def prepare_corpus():
    
    sentence_list = []
    
    regex = re.compile('[^a-zA-Z]+')
    sentences = brown.sents()
    
    for sent in sentences:
        sentence = ' '.join(sent)
        sentence = regex.sub(' ',sentence)
        sentence = re.sub(' +',' ',sentence)
        sentence = sentence.upper()
        sentence = '<s> '+sentence+'</s>'
        sentence_list.append(sentence)
    
    with open('data/brown.txt','w') as f:
        for sentence in sentence_list:
            f.write(sentence+'\n')


def process_TREC_questions(fle):
    
    q_list = []
    q_list_pos_tags=[]
    
    regex = re.compile('[^a-zA-Z0-9]+')
    with open(fle,'r') as f:
        for line in f:
        
            question = line.split(' ',1)[1]
            question = regex.sub(' ',question)
            question = re.sub(' +',' ',question)
            question = question.upper()
            question = '<s> '+question+'<\s>'
            
            
            tagged = nltk.tag.pos_tag(question.split(),tagset='universal')
            tagged = [x[1] for x in tagged]
            tagged = ' '.join(tagged)
            tagged = '<s> '+tagged+' <\s>'
            
            #q_list.append(question)
            q_list_pos_tags.append(tagged)
    
    
    with open('data/Q_TREC.txt','w') as f:
        for q in q_list:
            f.write(q+'\n')
    
                    
    with open('data/Q_TREC_pos.txt','w')as f:
        for line in q_list_pos_tags:
            f.write(line+'\n')


#less score is good
def by_score(question):
    score = question.split('\t')[-1]
    return score


def write_to_file(filename,qids,qid_to_question):
    
    with open(filename,'w') as f:
        for qid in qids:
            questions = qid_to_question[qid]
            questions = sorted(questions,key=by_score)
            #associate rank with each question based on its score
            for rank,q in enumerate(questions):
                q = q.rstrip('\n')
                q = q+'\t'+str(rank+1)+'\n'
                questions[rank] = q
                #f.write(q)
            
            random.shuffle(questions)
            
            for q in questions:
                f.write(q)


#randomly partitions the gold-standard data-set to produce
#training data-set and test data-set.
#this has some bias because the same qid used for training
#might be present in dev and test sets.
#Fix it, by sampling based on qids
def partition_data(gs,train,test,fr):
    
    
    with open(gs,'r') as f:
        data  = f.readlines()
        
    # a map from qid to its set of questions
    qid_to_question = defaultdict(list)
    
    for line in data:
        qid = int(line.split('\t')[0])
        qid_to_question[qid].append(line)
        
    qids = qid_to_question.keys()
        
    random.shuffle(qids)
    
    n = len(qids)
    
    idx = int(n*fr)
    
    tr = sorted(qids[0:idx])
    tt = sorted(qids[idx:2*idx])
    
    print tr
    print tt
    
    print 'Total no. of samples = '+str(n)
    print 'No. of training samples = '+str(idx)
    print 'No. of test samples = '+str(n-idx)
    
    write_to_file(train,tr,qid_to_question)
    write_to_file(test,tt,qid_to_question)

def print_GS_statistics(gs):
    
    with open(gs,'r') as f:
        data  = f.readlines()
        
    
    scoreList = defaultdict(float)
    
    for line in data:
        score = float(line.split('\t')[-1])
        scoreList[score]+=1 
    
    print sorted(scoreList.items(), key=operator.itemgetter(0))
        
if __name__ == "__main__":
    
    input_file = '/Users/sanchitagarwal/Documents/EclipseWork/QG/Ranker/data/QGSTEC-Sentences-2010/TestData_QuestionsFromSentences.xml'
    gold_standard = '/Users/sanchitagarwal/Documents/EclipseWork/QG/Ranker/data/gs.txt'
    qFile = './data/Heilman_questions.txt'
    test_heilman_file = './data/test_heilman.txt'
    trec = 'data/QuestionsTREC.txt'
    
    gold_standard = 'data/gs.txt'
    gold_standard_cleaned = 'data/gs_cleaned.txt'
    #train = 'data/train.txt'
    train = 'data/train_sample.txt'
    #test = 'data/test.txt'
    test = 'data/test_sample.txt'

    #extract_data(input_file,gold_standard)
    #clean_gs(gold_standard,gold_standard_cleaned)
    #prepareHeilmanQuestions(qFile,test_heilman_file)
    #prepare_corpus()
    #prepare_tagged_corpus()
    #process_TREC_questions(trec)
    #partition_data(gold_standard_cleaned,train,test,0.03)
    print_GS_statistics(gold_standard_cleaned)
    