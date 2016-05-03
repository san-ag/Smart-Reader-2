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
        sentence = '<s> '+sentence+' </s>'
        sentence = re.sub('\.+','',sentence)
        sentence = re.sub(' +',' ',sentence)
        tagged_list.append(sentence)
        
    
    with open('data/brown_POS.txt','w') as f:
        for item in tagged_list:
            f.write(item+'\n')
        
    #print V
    
def prepare_tagged_corpus_1():
    
    tagged_list = []
    
    with open('data/brown.txt','r') as f:
        data = f.read().splitlines()
        
    print data[0:2]
        
    for line in data:
        pos_tags = ''
        tagged = nltk.tag.pos_tag(line.split())
        #print tagged
    
        for item in tagged:
            if item[0] == ('<s>' or '</s>'):
                pos_tags+=item[0]+' '
            #elif item[1].startswith('NN'):
            #    pos_tags+='NN '
            #elif item[1].startswith('VB'):
            #    pos_tags+='VB '
            else:
                pos_tags+=item[1]+' '
        
        pos_tags  = pos_tags.rstrip()
        
        tagged_list.append(pos_tags)
            
    with open('data/brown_pos_detailed.txt','w') as f:
        for item in tagged_list:
            f.write(item)
            f.write('\n')

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
            
            
            tagged = nltk.tag.pos_tag(question.split())
            tagged = [x[1] for x in tagged]
            tagged = ' '.join(tagged)
            tagged = '<s> '+tagged+' <\s>'
            
            q_list.append(question)
            q_list_pos_tags.append(tagged)
    
    
    with open('data/Q_TREC.txt','w') as f:
        for q in q_list:
            f.write(q+'\n')
    
                    
    with open('data/Q_TREC_pos_detailed.txt','w')as f:
        for line in q_list_pos_tags:
            f.write(line+'\n')


#less score is good
def by_score(question):
    score = question.split('\t')[-1]
    return score


def write_to_file(filename,qids,qid_to_question,category):
    
    with open(filename,'w') as f:
        for qid in qids:
            questions = qid_to_question[qid]
            
            if category == 'ranking':
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
def partition_data(gs,train,test,fr,category):
    
    
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
    tt = sorted(qids[idx:])
    
    print tr
    print tt
    
    print 'Total no. of samples = '+str(n)
    print 'No. of training samples = '+str(idx)
    print 'No. of test samples = '+str(n-idx)
    
    write_to_file(train,tr,qid_to_question,category)
    write_to_file(test,tt,qid_to_question,category)

def print_GS_statistics(gs):
    
    with open(gs,'r') as f:
        data  = f.readlines()
        
    
    scoreList = defaultdict(float)
    
    for line in data:
        score = float(line.split('\t')[-1])
        scoreList[score]+=1 
    
    print sorted(scoreList.items(), key=operator.itemgetter(0))

def binarize(gs,gs_binary):
    
    count = 0
    
    with open(gs,'r') as f:
        data = f.read().splitlines()
    
    for idx,line in enumerate(data):
        parts = line.split('\t')
        score = parts[-1]
        if float(score)<2.5:
            score = '1'  #acceptable
            count+=1
        else: 
            score = '0'  #unacceptable
        data[idx] = ('\t'.join(parts[0:-1]))+'\t'+score+'\n'
        
    with open(gs_binary,'w') as f:
        for line in data:
            f.write(line)
        
    print count
    
def process_annotated_questions(original,rated1,rated2):
    
    context_question_dict = defaultdict(list)
    
    question_rating1_dict = {}
    question_rating2_dict = {}
    question_rating_mean_dict = {}
    
    final = []
    
    with open(original,'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            q = parts[0]
            c = parts[1]
            context_question_dict[c].append(q)
        
    with open(rated1,'r') as f:
        for line in f:
            line = line.rstrip()
            #print line
            parts = line.split(' ',1)
            q = parts[1]
            rating = float(parts[0])
            question_rating1_dict[q] = rating
            
    with open(rated2,'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split(' ',1)
            q = parts[1]
            rating = float(parts[0])
            question_rating2_dict[q] = rating
    
    #average rating
    ques = question_rating1_dict.keys()
    for q in ques:
        question_rating_mean_dict[q] = (question_rating1_dict[q]+question_rating2_dict[q])/2
    
            
    contexts = context_question_dict.keys()
    
    for idx,c in enumerate(contexts):
        ques = context_question_dict[c]
        for q in ques:
            rating = question_rating_mean_dict[q]
            line = str(idx+1)+'\t'+c+'\t'+q+'\t'+str(rating)+'\n'
            final.append(line)
            
    with open('data/test_new.txt','w') as f:
        for line in final:
            f.write(line)
        
if __name__ == "__main__":
    
    input_file = '/Users/sanchitagarwal/Documents/EclipseWork/QG/Ranker/data/QGSTEC-Sentences-2010/TestData_QuestionsFromSentences.xml'
    gold_standard = '/Users/sanchitagarwal/Documents/EclipseWork/QG/Ranker/data/gs.txt'
    qFile = './data/Heilman_questions.txt'
    test_heilman_file = './data/test_heilman.txt'
    trec = 'data/QuestionsTREC.txt'
    
    gold_standard = 'data/gs.txt'
    gold_standard_cleaned = 'data/gs_cleaned.txt'
    gold_standard_binary = 'data/gs_binary.txt'
    
    train = 'data/train.txt'
    #train = 'data/train_sample.txt'
    test = 'data/test.txt'
    #test = 'data/test_sample.txt'
    
    original = 'data/QGSTEC2010-TaskB-MH-QG-500outputs.txt'
    rated_yu = 'data/questions_YU.txt'
    rated_sh = 'data/questions_SH.txt'
    
    #prepareHeilmanQuestions(qFile,test_heilman_file)
    #prepare_corpus()
    #prepare_tagged_corpus()
    #process_TREC_questions(trec)
    #prepare_tagged_corpus_1()
    
    process_annotated_questions(original,rated_yu,rated_sh)

    #extract_data(input_file,gold_standard)
    #clean_gs(gold_standard,gold_standard_cleaned)
    #binarize(gold_standard_cleaned,gold_standard_binary)
    
    #partition_data(gold_standard_cleaned,train,test,0.03,'ranking')
    #partition_data(gold_standard_binary,train,test,0.80,'binary_labels')
    
    #print_GS_statistics(gold_standard_cleaned)
    