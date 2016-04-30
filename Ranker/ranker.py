import sys
import model as md
import random
from _collections import defaultdict

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
    
    #use 'fr' fraction of data for training 
    #fr = 0.85
    
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
    #tt = sorted(qids[idx:])
    tt = sorted(qids[idx:2*idx])
    
    print tr
    print tt
    
    print 'Total no. of samples = '+str(n)
    print 'No. of training samples = '+str(idx)
    print 'No. of test samples = '+str(n-idx)
    
    write_to_file(train,tr,qid_to_question)
    write_to_file(test,tt,qid_to_question)

def main(arguments):
    
    if len(arguments)<2:
        print("usage: rank.py training_data test_data outputFile")
        
    training_data_file = arguments[0]
    test_data_file = arguments[1] 
    output_file = arguments[2]
    
    #md.trainRanker(training_data_file)
    md.testRanker(test_data_file)
    md.rank(test_data_file,output_file)
    md.evaluate(test_data_file,output_file)
    
if __name__ == "__main__":
    
    gold_standard = 'data/gs.txt'
    train = 'data/train.txt'
    #train = 'data/train_sample.txt'
    test = 'data/test.txt'
    #test = 'data/test_sample.txt'
    output = 'result/output.txt'
    
    #partition_data(gold_standard,train,test,0.03)
    
    main([train,test,output])